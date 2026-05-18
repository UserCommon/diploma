# 3. Реализация системы

\ 

## 3.1. Реализация сервиса обработки деталей

\ 

Сервис `accessories_processing` отвечает за управление каталогом тюнинговых деталей и реализован на FastAPI с асинхронной очередью SAQ. Центральная инженерная задача сервиса — интеграция тяжёлых ML-моделей (rembg, sentence-transformers) в полностью асинхронный стек без блокировки основного event loop.

**Проблема запуска ML-моделей в асинхронном контексте**

Большинство ML-библиотек Python — rembg, sentence-transformers, numpy — реализованы в синхронном стиле и не поддерживают asyncio. Прямой вызов таких функций из async-обработчика заблокировал бы event loop FastAPI на всё время вычисления, делая сервер недоступным для других запросов в течение нескольких секунд. Стандартное решение — перемещение тяжёлых вычислений в отдельный поток через `loop.run_in_executor`, что позволяет event loop продолжать обрабатывать остальные запросы, пока модель считает в фоновом потоке.

Вторая проблема — эффективное управление жизненным циклом модели. Загрузка ML-модели с диска занимает несколько секунд и требует значительных объёмов памяти. Повторная инициализация при каждом запросе недопустима. Решение — паттерн ленивого синглтона на основе декоратора `@lru_cache(maxsize=1)`: функция-фабрика вызывается один раз при первом обращении и кэшируется Python на всё время жизни процесса.

**Удаление фона**

Обработка фотографий деталей реализована в модуле `accessories_processing/app/models/rembg_model.py`. Библиотека rembg использует нейросетевую модель U2Net, предобученную на задаче матирования (image matting) — выделения объекта переднего плана с точностью до полупрозрачных краёв. Объект `Session`, создаваемый через `rembg.new_session("u2net")`, инкапсулирует загруженные веса модели и контекст ONNX Runtime. Создание сессии — ресурсоёмкая операция; повторное использование одного объекта через `lru_cache` исключает накладные расходы при каждом вызове:

```python
@lru_cache(maxsize=1)
def _get_session():
    return rembg.new_session("u2net")

async def remove_background(image_bytes: bytes) -> bytes:
    loop = asyncio.get_running_loop()
    session = _get_session()
    result = await loop.run_in_executor(
        None, lambda: rembg.remove(image_bytes, session=session)
    )
    return result
```

Аргумент `None` в `run_in_executor` означает использование стандартного `ThreadPoolExecutor`, который FastAPI создаёт и управляет автоматически. Лямбда-функция замыкает уже инициализированную сессию, передавая её в поток без повторной инициализации.

**Вычисление векторных эмбеддингов**

Модуль `accessories_processing/app/services/embeddings.py` реализует кодирование текстовых описаний деталей в векторные представления. Применяется тот же паттерн `lru_cache + run_in_executor`:

```python
@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

async def embed_text(text: str) -> list[float]:
    loop = asyncio.get_running_loop()
    model = _get_model()
    vector = await loop.run_in_executor(None, lambda: model.encode(text))
    return vector.tolist()
```

Функция `model.encode()` принимает строку и возвращает numpy-массив из 384 чисел с плавающей точкой. Метод `.tolist()` преобразует его в обычный список Python для последующей сериализации в JSON и сохранения в PostgreSQL через pgvector.

**Жизненный цикл детали: задача prepare_part**

При загрузке новой детали через `POST /api/parts/upload` API сохраняет исходное изображение на диск, создаёт запись в БД со статусом `pending` и немедленно возвращает клиенту `task_id`. Фактическая обработка выполняется асинхронно воркером SAQ в задаче `prepare_part`:

```python
async def prepare_part(ctx: dict, *, part_id: str) -> None:
    # 1. Удалить фон
    image_bytes = image_path.read_bytes()
    processed_bytes = await remove_background(image_bytes)

    # 2. Сохранить обработанное изображение
    dest.write_bytes(processed_bytes)

    # 3. Вычислить эмбеддинг и перевести в статус ready
    text = f"{part.name} {part.description or ''}".strip()
    part.embedding = await embed_text(text)
    part.processed_image_url = processed_url
    part.status = "ready"
    await db.commit()
```

Три операции — удаление фона, сохранение файла, вычисление эмбеддинга — выполняются строго последовательно: каждая следующая зависит от результата предыдущей. При ошибке на любом шаге запись переводится в статус `failed`, что позволяет администратору идентифицировать проблемные загрузки.

**Семантический поиск с косинусным сходством**

Реализация поиска в `accessories_processing/app/api/parts.py` вычисляет сходство в Python через numpy, что даёт полную переносимость между базами данных. Запрос кодируется в вектор той же моделью `all-MiniLM-L6-v2`, затем для каждой детали со статусом `ready` вычисляется косинусное сходство:

```python
query_vec = await embed_text(body.query)
q = np.array(query_vec)

def cosine_sim(part: Part) -> float:
    if part.embedding is None:
        return -1.0
    v = np.array(part.embedding)
    denom = np.linalg.norm(q) * np.linalg.norm(v)
    return float(np.dot(q, v) / denom) if denom else 0.0

min_score = 0.4
scored = sorted(((cosine_sim(p), p) for p in parts), reverse=True)
items = []
for score, p in scored[:body.limit]:
    if score < min_score:
        break
    ...
```

Порог 0.4 выбран экспериментально: при более низком пороге в результаты попадают семантически далёкие детали (например, по запросу «карбоновый спойлер» возвращается «резиновый коврик»), при более высоком — отсекаются синонимичные формулировки. Детали возвращаются отсортированными по убыванию релевантности; агент получает поле `score` и может учитывать его при формулировке промпта генерации.

\ 

## 3.2. Реализация сегментационного пайплайна

\ 

Сегментационный пайплайн реализован в `core_service/app/models/segmentation.py`. Архитектурно он решает три задачи: детекцию объекта по тексту, построение точной пиксельной маски и корректное управление тяжёлыми моделями в асинхронном сервисе.

**Ленивые синглтоны и выбор устройства**

SAM 2 и GroundingDINO хранятся в модульных переменных и инициализируются при первом обращении. Такой подход — в отличие от `lru_cache` — позволяет использовать `global` для явного управления состоянием и упрощает предзагрузку при старте воркера:

```python
_sam2_predictor: Any = None
_gdino_model: Any = None

def _load_sam2() -> Any:
    global _sam2_predictor
    if _sam2_predictor is not None:
        return _sam2_predictor
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = _get_device()
    model = build_sam2(settings.SAM2_CONFIG, settings.SAM2_CHECKPOINT, device=device)
    _sam2_predictor = SAM2ImagePredictor(model)
    return _sam2_predictor
```

Функция `_get_device()` выбирает CUDA при наличии GPU, иначе возвращает CPU — Apple MPS явно исключён, поскольку часть операций SAM 2 и GroundingDINO не поддерживается MPS-бэкендом PyTorch.

**Предзагрузка при старте воркера**

SAM 2 и GroundingDINO должны загружаться в строго определённом порядке: совместная инициализация `torch.jit` при обратном порядке приводит к конфликту контекстов. Воркер предзагружает обе модели при запуске, до начала обработки задач:

```python
async def _preload_models() -> None:
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _load_sam2)
        await loop.run_in_executor(None, _load_gdino)
        logger.info("Models pre-loaded successfully.")
    except Exception as exc:
        logger.warning("Model pre-load skipped: %s", exc)

async def main() -> None:
    await _preload_models()
    queue = await get_queue()
    worker = saq.Worker(queue, functions=[run_agent_session], concurrency=2)
    await worker.start()
```

Исключение при предзагрузке обрабатывается мягко (`warning`, не `error`): это позволяет запускать сервис в среде разработки без установленных зависимостей SAM 2/GroundingDINO — в таком случае активируется заглушка.

**Препроцессинг в GroundingDINO**

GroundingDINO ожидает изображение в виде нормализованного тензора 800×800. Функция `_get_gdino_boxes` выполняет предобработку через torchvision transforms и конвертирует результирующие координаты из нормализованного формата (cx, cy, w, h) в абсолютные пиксельные координаты (x1, y1, x2, y2), необходимые SAM 2:

```python
transform = T.Compose([
    T.Resize((800, 800)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

boxes_cx, _logits, _phrases = predict(
    model=model, image=img_tensor, caption=text,
    box_threshold=0.35, text_threshold=0.25, device=device,
)

for box in boxes_cx.tolist():
    cx, cy, bw, bh = box
    result.append([
        (cx - bw / 2) * w, (cy - bh / 2) * h,
        (cx + bw / 2) * w, (cy + bh / 2) * h,
    ])
```

Порог `box_threshold=0.35` отсекает детекции с низкой уверенностью модели, `text_threshold=0.25` — отдельный порог для соответствия текстового токена визуальному региону. Оба значения подобраны экспериментально на реальных фотографиях автомобилей.

**Пайплайн сегментации и объединение масок**

Функция `_segment_sync` реализует полный пайплайн синхронно — она вызывается через `run_in_executor` из асинхронного контекста:

```python
def _segment_sync(image: Image.Image, text: str) -> Image.Image:
    try:
        import groundingdino
        import sam2
    except ImportError:
        return _stub_mask(image)

    boxes = _get_gdino_boxes(image, text)
    if not boxes:
        return Image.new("L", image.size, 0)

    predictor = _load_sam2()
    predictor.set_image(np.array(image.convert("RGB")))

    combined = np.zeros(image.size[::-1], dtype=np.uint8)
    for box in boxes:
        box_np = np.array(box, dtype=np.float32)
        masks, _, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=box_np[None, :], multimask_output=False,
        )
        combined = np.maximum(combined, masks[0].astype(np.uint8) * 255)

    return Image.fromarray(combined, mode="L")
```

GroundingDINO может вернуть несколько bounding box'ов — например, для запроса «колёса» он обнаружит все четыре (или видимые два-три) колеса на фото. SAM 2 строит отдельную маску для каждого box'а. Итоговая маска формируется как поэлементный максимум всех частных масок (`np.maximum`): область, отмеченная хотя бы одной из масок, становится белой (255). Это обеспечивает корректную обработку множественных вхождений одного объекта без специальных ветвлений в логике.

Параметр `multimask_output=False` указывает SAM 2 возвращать одну наиболее вероятную маску вместо трёх кандидатов — этого достаточно при наличии точного bounding box от детектора.

**Заглушка для среды без GPU**

Если пакеты `sam2` или `groundingdino` не установлены, функция возвращает центральный прямоугольник 50×50% изображения. Это намеренный компромисс: заглушка даёт нереалистичную маску, но позволяет полностью тестировать остальные части системы — агента, провайдеров генерации, SSE-стриминг — без GPU-инфраструктуры:

```python
def _stub_mask(image: Image.Image) -> Image.Image:
    from PIL import ImageDraw
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle(
        [int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)],
        fill=255,
    )
    logger.warning("Using stub centre-box mask (SAM2/GroundingDINO not installed).")
    return mask
```

\ 

## 3.3. Реализация агента и оркестрации

\ 

Агентная система является центральным компонентом `core_service`. Она отвечает за приём пользовательского запроса, управление последовательностью вызовов инструментов и сохранение хода выполнения для стриминга клиенту.

**Системный промпт и предотвращение галлюцинаций**

Ключевая инженерная проблема при использовании LLM в качестве оркестратора пайплайна — склонность модели добавлять в промпт генерации визуальные детали из своих обучающих данных. Модель «знает», как выглядят диски Volk Racing TE37, и может добавить в промпт «золотые шестиспицевые диски», даже если в каталоге хранится экземпляр совершенно другого цвета. Результат — сгенерированное изображение не соответствует реальной детали из каталога.

Для решения проблемы в системном промпте сформулирован явный запрет с примером:

```python
_DEFAULT_SYSTEM = (
    "You are an AI car tuning assistant. Help users visually tune cars by:\n"
    "1. Searching the parts library for relevant components\n"
    "2. Segmenting the target area on the car to create a mask\n"
    "3. Generating an inpainted result with the new part applied\n\n"
    "CRITICAL RULE for generate_image prompt:\n"
    "Use ONLY the exact part name and description returned by search_parts.\n"
    "Do NOT invent, add, or infer any visual details (color, finish, material, "
    "spoke count, etc.) that are not explicitly stated in the search results.\n"
    "Do NOT use your own knowledge about what that brand/model looks like.\n"
    "Example: if search returns name='Volk Racing TE37' with no color info, "
    "the prompt must be 'Replace wheels with Volk Racing TE37, photorealistic, "
    "matching car lighting' — nothing more.\n\n"
    "If multiple car images are provided, pass the first as source_image_url "
    "and the rest as extra_source_image_urls (comma-separated).\n\n"
    "Think step by step. Use tools in order."
)
```

Промпт содержит конкретный пример допустимого и недопустимого поведения, что значительно более эффективно, чем абстрактный запрет — языковые модели лучше следуют инструкциям с образцом.

**Поддержка нескольких LLM-провайдеров**

Архитектура агента поддерживает три провайдера языковых моделей, что позволяет использовать систему как с коммерческими API (OpenAI, Anthropic), так и с локально развёрнутыми моделями через OpenAI-совместимый интерфейс (Ollama, LM Studio). Выбор провайдера осуществляется через переменную окружения `LLM_PROVIDER` без изменения кода:

```python
def _build_llm() -> Any:
    provider = settings.LLM_PROVIDER.lower()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            temperature=0.2,
        )
    if provider == "openai_compatible":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL or None,
            temperature=0.2,
            request_timeout=300,
        )
    # default: openai
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        temperature=0.2,
        request_timeout=300,
    )
```

Температура `0.2` обеспечивает детерминированный выбор инструментов — более высокие значения приводят к нестабильному порядку вызовов. Таймаут `300` секунд установлен с учётом того, что за одну сессию модель может сделать несколько вызовов инструментов, каждый из которых занимает десятки секунд (сегментация, генерация изображения).

**Режим заглушки**

При отсутствии `LLM_PROVIDER` в переменных окружения агент переключается в режим заглушки `_run_stub_agent`, который напрямую вызывает инструменты в фиксированном порядке, минуя языковую модель. Это позволяет тестировать пайплайн в целом без LLM API ключа — что критически важно для локальной разработки.

**Жизненный цикл сессии в воркере**

Задача `run_agent_session` выполняется воркером SAQ. Она обновляет статус сессии на `processing`, запускает агент и сохраняет каждое событие в поле `events` записи БД по мере поступления:

```python
async for event in run_agent_stream(...):
    async with async_session() as db:
        s = await db.get(AgentSession, sid)
        s.events = list(s.events or []) + [event]
        if event.get("type") == "result":
            s.result_image_urls = event.get("result_image_urls", [])
        await db.commit()
```

Каждое событие фиксируется в отдельной транзакции — это гарантирует, что SSE-стрим получит событие немедленно после его генерации агентом, даже если последующие шаги завершатся ошибкой.

**SSE-стриминг через опрос базы данных**

Эндпоинт `GET /api/agent/sessions/{id}/stream` реализует SSE через периодический опрос БД с интервалом 0.5 секунды. Переменная `seen` хранит количество уже отправленных событий, что позволяет отправлять клиенту только новые:

```python
async def generator() -> AsyncGenerator[str, None]:
    seen = 0
    while True:
        async with async_session() as db:
            s = await db.get(AgentSession, uuid.UUID(session_id))
            if s is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'session not found'})}\n\n"
                return
            for event in (s.events or [])[seen:]:
                yield f"data: {json.dumps(event)}\n\n"
            seen = len(s.events or [])
            if s.status in ("complete", "failed"):
                yield f"data: {json.dumps({'type': 'done', 'status': s.status})}\n\n"
                return
        await asyncio.sleep(0.5)
```

Архитектурное решение — опрос БД вместо прямой передачи событий от воркера к SSE-обработчику — обусловлено тем, что воркер и API работают в разных процессах (и потенциально на разных машинах). Прямая передача потребовала бы дополнительного брокера сообщений (Redis Pub/Sub или WebSocket-сервера). Опрос БД решает задачу без усложнения инфраструктуры: любой экземпляр API может обслуживать стрим любой сессии, читая данные из общей БД.

\ 

## 3.4. Реализация провайдеров инпейнтинга

\ 

Подсистема генерации изображений спроектирована по паттерну «стратегия»: конкретный алгоритм генерации инкапсулирован в отдельном классе-провайдере, а агент работает с унифицированным интерфейсом, не зависящим от реализации.

**Протокол InpaintProvider**

Интерфейс провайдера определён через Python `Protocol` — механизм структурной типизации, не требующий наследования от базового класса. Это упрощает добавление новых провайдеров: достаточно реализовать единственный асинхронный метод `generate`:

```python
@runtime_checkable
class InpaintProvider(Protocol):
    async def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        n: int = 1,
        part_image: Image.Image | None = None,
        extra_images: list[Image.Image] | None = None,
    ) -> list[Image.Image]: ...
```

Параметр `part_image` передаёт изображение детали из каталога — провайдеры, поддерживающие референсные изображения (Polza/FLUX.2 Pro), используют его для максимально точного воспроизведения внешнего вида детали. Параметр `extra_images` — список дополнительных фото автомобиля с разных ракурсов, которые FLUX.2 Pro учитывает для понимания геометрии автомобиля.

**Фабрика провайдеров**

Функция `_make_provider` реализует выбор провайдера по строковому имени. Провайдер может быть задан через переменную окружения `INPAINT_PROVIDER` (для всего сервиса) или передан в параметре конкретного запроса (для переключения из UI без перезапуска сервиса):

```python
def _make_provider(name: str) -> InpaintProvider:
    n = name.lower()
    if n == "polza":     return PolzaInpaintProvider()
    if n == "gpt_image": return GptImageInpaintProvider()
    if n == "diffusers": return DiffusersFluxFillProvider()
    if n == "sdxl":      return DiffusersSDXLInpaintProvider()
    if n == "genapi":    return GenApiFluxInpaintProvider()
    if n == "kontext":   return GenApiKontextProvider()
    if n == "klein":     return GenApiKleinProvider()
    return StubInpaintProvider()
```

**Локальные провайдеры: SDXL и FLUX.1 Fill**

Оба локальных провайдера загружают веса через библиотеку `diffusers` и запускаются на доступном устройстве. `DiffusersSDXLInpaintProvider` использует модель `stable-diffusion-xl-1.0-inpainting-0.1` (~7 ГБ), не требующую токена HuggingFace. Перед подачей в модель маска дилатируется фильтром `MaxFilter(15)` — расширение маски на 15 пикселей даёт модели контекст вокруг границ объекта, что заметно улучшает качество стыковки сгенерированного региона с оригинальным изображением:

```python
mask = mask.filter(ImageFilter.MaxFilter(15))
result = pipe(
    prompt=prompt, image=image, mask_image=mask,
    strength=0.99, padding_mask_crop=32,   # кадрировать вокруг маски
    guidance_scale=7.5, num_inference_steps=30,
)
```

`DiffusersFluxFillProvider` загружает FLUX.1 Fill Dev (~20 ГБ). FLUX требует размеры изображения, кратные 8 — это техническое ограничение архитектуры трансформерных блоков, которое проверяется и исправляется перед вызовом:

```python
w = (w // 8) * 8
h = (h // 8) * 8
image = image.resize((w, h), Image.LANCZOS)
mask  = mask.resize((w, h), Image.LANCZOS)
```

Оба локальных провайдера запускают синхронный пайплайн диффузии через `run_in_executor`, параллельно генерируя `n` вариантов через `asyncio.gather`.

**Главный провайдер: Polza (FLUX.2 Pro)**

`PolzaInpaintProvider` взаимодействует с API polza.ai, предоставляющим доступ к FLUX.2 Pro — модели с 32 млрд параметров и нативной поддержкой массива входных изображений. Ключевая особенность реализации — нестандартный порядок передачи изображений, который позволяет модели самостоятельно интерпретировать роли входных данных на основе промпта:

```python
# Порядок: [0] маска, [1] референс детали (если есть), [2] фото авто, [3+] доп. ракурсы
images_payload = [{"type": "base64", "data": _img_to_b64(mask_resized)}]
if part_image is not None:
    ref = part_image.convert("RGB").copy()
    ref.thumbnail((1024, 1024), Image.LANCZOS)
    images_payload.append({"type": "base64", "data": _img_to_b64(ref)})
images_payload.append({"type": "base64", "data": _img_to_b64(img_resized)})
for extra in (extra_images or []):
    images_payload.append({"type": "base64", "data": _img_to_b64(ex)})
```

Промпт явно указывает модели роль каждого изображения через нумерацию, что является рекомендованным подходом для мультиизображённых запросов к FLUX.2 Pro:

```python
final_prompt = (
    f"{prompt}. "
    "Use image 1 as the inpainting mask. "
    "Use image 2 as the reference part — replicate its exact visual appearance, "
    "color, and finish. "
    f"Edit car photo in image {car_index}."
)
```

API polza.ai может возвращать результат либо синхронно, либо в виде задачи с `job_id`. Провайдер обрабатывает оба случая: при синхронном ответе URL извлекается немедленно, при асинхронном запускается цикл опроса с интервалом 2 секунды и таймаутом 180 секунд.

Генерация `n` вариантов запускается параллельно через `asyncio.gather`; каждый вариант получает уникальный seed (базовый 1000, шаг 42), что обеспечивает разнообразие результатов при детерминированной воспроизводимости:

```python
async def generate(self, image, mask, prompt, n=1, ...) -> list[Image.Image]:
    tasks = [
        _call_api(image, mask, prompt, seed=1000 + i * 42, ...)
        for i in range(n)
    ]
    return list(await asyncio.gather(*tasks))
```

\ 

## 3.5. Реализация клиентского приложения

\ 

Клиентское приложение построено на SvelteKit и организовано в две основные страницы: Studio — интерактивная рабочая поверхность для запуска агента, и Parts — каталог деталей. Приложение взаимодействует с бэкендом исключительно через REST JSON API и SSE-стрим; состояние хранится реактивно в компонентах Svelte без внешних state management библиотек.

**Studio: управление загрузкой изображений**

Страница Studio поддерживает загрузку нескольких фотографий автомобиля — с разных ракурсов для улучшения понимания геометрии моделью FLUX.2 Pro. Изображения принимаются как через стандартный диалог выбора файла, так и через drag-and-drop. Миниатюры предварительного просмотра создаются через `URL.createObjectURL` без отправки файлов на сервер:

```javascript
function handleFiles(files) {
    images = [...images, ...Array.from(files)];
    previews = images.map(f => URL.createObjectURL(f));
}
```

Интерфейс позволяет удалять отдельные изображения из очереди до отправки. Пользователь выбирает провайдера генерации и количество вариантов результата прямо в UI — эти параметры передаются в API как поля формы и маршрутизируются до воркера без изменения логики агента.

**Запуск агента и подключение к SSE-стриму**

Функция `start()` формирует `multipart/form-data` и отправляет запрос к бэкенду. После получения `session_id` немедленно открывается SSE-соединение. Каждое входящее событие добавляется в реактивный массив `events`, что автоматически обновляет лог без явных DOM-манипуляций — реактивность Svelte перерисовывает только изменившиеся части компонента:

```javascript
async function start() {
    const fd = new FormData();
    fd.append('user_prompt', prompt);
    fd.append('n', String(nVariants));
    fd.append('inpaint_provider', inpaintProvider);
    for (const img of images) fd.append('images', img);

    const { session_id } = await createSession(fd);
    sessionId = session_id;

    stopStream = streamSession(
        session_id,
        (event) => {
            events = [...events, event];
            if (event.type === 'result' && event.result_image_urls?.length) {
                resultUrls = event.result_image_urls;
                selectedResult = resultUrls[0];
            }
        },
        (finalStatus) => {
            status = finalStatus;
            stopStream = null;
        }
    );
}
```

Функция `streamSession` возвращает функцию `stopStream`, вызов которой закрывает EventSource и прекращает опрос. Это необходимо для корректного сброса состояния при повторном запуске агента.

**Компонент EventLog**

`EventLog.svelte` отображает журнал выполнения агента в монопространственном шрифте с цветовой дифференциацией по типу события. Компонент принимает массив `events` через `export let events`, что делает его чисто презентационным без собственного состояния:

```svelte
{#each events as event}
  <div class="entry type-{event.type}">
    <span class="icon">{icons[event.type] ?? '·'}</span>
    <div class="body">
      {#if event.type === 'thinking'}
        <span class="muted">{event.content}</span>
      {:else if event.type === 'tool_call'}
        <span class="tool">{event.tool}</span>
        <span class="muted"> ← {event.input}</span>
      {:else if event.type === 'tool_result'}
        <span class="muted output">{event.output?.slice(0, 200)}{event.output?.length > 200 ? '…' : ''}</span>
      {:else if event.type === 'result'}
        <span>{event.output}</span>
      {:else if event.type === 'error'}
        <span class="error">{event.message}</span>
      {/if}
    </div>
  </div>
{/each}
```

Стили привязываются к типу события через динамический CSS-класс `type-{event.type}`: мысли агента (`thinking`) отображаются приглушённым серым цветом, вызовы инструментов (`tool_call`) — на тёмно-синем фоне с выделенным именем инструмента, результаты инструментов (`tool_result`) — на нейтральном фоне, финальный результат (`result`) — на тёмно-зелёном. Длинные выводы инструментов обрезаются до 200 символов с многоточием, чтобы не перегружать интерфейс объёмными JSON-ответами.

**Страница Parts**

Страница каталога деталей отображает сетку карточек `PartCard.svelte` с поддержкой фильтрации по домену и категории. Карточки показывают обработанное изображение (с прозрачным фоном на тёмном фоне карточки), название, категорию и статус обработки. Детали со статусом `pending` и `failed` отображаются визуально отличительно, что позволяет администратору контролировать процесс обработки каталога непосредственно из браузера.

\ 

## 3.6. Тестирование и оценка качества системы

\ 

### 3.6.1. Методология тестирования

Тестирование системы проводилось по трём направлениям: оценка качества семантического поиска по каталогу, оценка качества сегментации изображений и оценка качества генерации результирующих изображений. Для объективной оценки генерации применялась метрика CLIP Score — косинусное сходство между эмбеддингами изображения и текстового промпта, вычисляемое моделью CLIP ViT-B/32. Значение CLIP Score находится в диапазоне [0, 1]; для фотореалистичных генераций характерен диапазон 0,15–0,30. Тестирование проводилось на реальных данных системы, накопленных в ходе разработки: 73 сессии, 41 из которых завершилась с результирующим изображением.

### 3.6.2. Оценка семантического поиска

Для оценки семантического поиска сформирован набор из 12 тестовых запросов: 10 релевантных (ожидаемый результат известен) и 2 заведомо нерелевантных («тонировка стёкла», «межпланетный корабль»). Запросы разделены на две группы: на английском языке (соответствует языку каталога) и на русском языке (кросс-языковой сценарий).

| Запрос | Язык | Ожидаемый результат | P@1 | Score |
|---|---|---|---|---|
| spoiler | EN | GT WING SPOILER | ✓ | 0,538 |
| rear wing carbon spoiler | EN | GT WING SPOILER | ✓ | 0,594 |
| gt wing | EN | GT WING SPOILER | ✓ | 0,651 |
| wheels rims sport | EN | volk_racing_1 | ✓ | 0,646 |
| sport rims | EN | volk_racing_1 | ✓ | 0,584 |
| volk racing | EN | volk_racing_1 | ✓ | 0,688 |
| headlights audi | EN | audi_a4 | ✓ | 0,814 |
| антикрыло спойлер | RU | GT WING SPOILER | ✗ | — |
| диски колёса | RU | volk_racing_1 | ✗ | — |
| фары ауди | RU | audi_a4 | ✗ | — |
| тонировка стёкла | RU | (нет) | ✓ | — |
| межпланетный корабль | RU | (нет) | ✓ | — |

По результатам тестирования: Precision@1 для англоязычных запросов составила **0,70**, Precision@3 — **0,70**. Среднее время ответа поискового эндпоинта — **25 мс**. Оценки релевантности для найденных деталей варьировались от 0,538 до 0,814, что соответствует семантически близким совпадениям.

Для русскоязычных запросов Precision@1 составила **0,00** — все три запроса вернули пустой результат. Анализ сырых оценок косинусного сходства показал, что проблема заключается не в отсутствии семантического понимания, а в недостаточности оценок для преодоления порога 0,4. Так, запрос «диски колёса» получил оценку 0,21 для детали `volk_racing_1` — модель корректно определила семантическую близость, однако значение оказалось вдвое ниже порогового. Для запроса «фары ауди» оценка не превысила 0,20, «антикрыло спойлер» — аналогично.

Это характерное ограничение кросс-языкового режима модели `all-MiniLM-L6-v2`: в рамках одного языка оценки для релевантных пар составляют 0,54–0,81; при кросс-языковом сопоставлении (русский запрос — английский каталог) те же семантически близкие пары дают 0,21 и ниже. Порог 0,4 оптимален для одноязычного поиска, но отсекает релевантные результаты в кросс-языковом сценарии.

Для устранения этого ограничения рекомендуется одно из двух решений: снизить порог до 0,25–0,30 при добавлении русскоязычных описаний в каталог, либо заменить модель на специализированную многоязычную, например `paraphrase-multilingual-MiniLM-L12-v2`, которая обучалась на параллельных корпусах и обеспечивает более высокие оценки при кросс-языковом поиске.

Нерелевантные запросы («тонировка стёкла», «межпланетный корабль») корректно возвращали пустой результат, что подтверждает правильность выбранного порогового значения для фильтрации шума.

### 3.6.3. Оценка качества сегментации

Для оценки сегментации рассмотрены маски, сгенерированные пайплайном GroundingDINO → SAM 2 в ходе реальных сессий. Анализ показал выраженную зависимость качества сегментации от семантической чёткости целевой области.

**Сегментация дисков** (текстовый запрос: `"wheel disc area"`): GroundingDINO корректно локализовал три видимых колеса автомобиля, SAM 2 построил точные маски с чёткими границами по контуру каждого диска. Результат соответствует задаче — область редактирования ограничена исключительно колёсами без захвата кузова или фона.

**Сегментация зоны спойлера** (текстовый запрос: `"rear trunk spoiler area"`): GroundingDINO вернул bounding box, охватывающий весь кузов автомобиля, а не только зону крышки багажника. SAM 2 построил маску всего автомобиля. Несмотря на избыточность маски, результирующее изображение оказалось приемлемым: FLUX.2 Pro, получив маску всего кузова и референсное изображение детали, корректно локализовал установку спойлера на крыше благодаря явному указанию ролей изображений в промпте.

Данное наблюдение выявляет ограничение системы: для объектов с выраженными визуальными границами (колёса, фары) сегментация работает точно, тогда как для абстрактных зон применения (крышка багажника, область порогов) детектор склонен к избыточному захвату. В таких случаях качество финального результата обеспечивается не точностью маски, а семантическим пониманием FLUX.2 Pro контекста задачи.

### 3.6.4. Оценка качества генерации (CLIP Score)

CLIP Score вычислялся как косинусное сходство между эмбеддингами результирующего изображения и описательного промпта, кодируемыми моделью CLIP ViT-B/32. Оценивались 5 изображений из 3 сценариев:

| Сценарий | Промпт оценки | CLIP Score |
|---|---|---|
| GT wing spoiler | a car with GT wing spoiler on the trunk | 0,1860 |
| Диски Volk Racing (сессия 1) | a car with Volk Racing sport wheels | 0,2197 |
| Диски Volk Racing (сессия 2) | a car with Volk Racing sport wheels | 0,2246 |
| Диски Volk Racing (сессия 3) | a car with Volk Racing sport wheels | 0,2256 |
| Диски Replay Audi A57 | a car with Replay Audi A57 alloy wheels | 0,1215 |

Средний CLIP Score по тестовой выборке составил **0,1955** (диапазон: 0,1215–0,2256). Наилучший результат показал сценарий замены дисков Volk Racing (0,222–0,226), что объясняется хорошей различимостью спортивных дисков в пространстве признаков CLIP. Наименьший результат — для дисков Replay Audi A57 (0,1215): тонкие визуальные отличия между оригинальными и заменёнными многоспицевыми дисками слабо отражаются в CLIP-эмбеддингах, поскольку общая сцена (автомобиль на белом фоне) остаётся неизменной.

Полученные значения CLIP Score находятся в диапазоне, характерном для задач тонкого редактирования изображений (0,15–0,25), где изменяется только локальный элемент сцены. Стандартный CLIP не оптимизирован для оценки подобных задач — он лучше различает глобальные изменения стиля, чем замену отдельной детали. Тем не менее стабильность результатов (три независимых запуска для Volk Racing дали 0,2197–0,2256) свидетельствует о воспроизводимости системы.

### 3.6.5. Статистика надёжности пайплайна

За период тестирования система обработала 73 сессии. Распределение по статусам и результатам приведено в таблице.

| Показатель | Значение |
|---|---|
| Всего сессий | 73 |
| Завершено успешно (status=complete) | 69 (94,5%) |
| Завершено с ошибкой (status=failed) | 2 (2,7%) |
| Сессий с результирующим изображением | 41 (56,2%) |
| Уникальных пользовательских запросов | 23 |
| Событий в типичной успешной сессии | 8 |

Разрыв между долей завершённых сессий (94,5%) и долей сессий с изображением (56,2%) объясняется двумя причинами: во-первых, часть сессий завершалась корректным сообщением агента об отсутствии подходящей детали в каталоге (без генерации изображения); во-вторых, часть ранних сессий выполнялась в режиме заглушки без подключённого провайдера генерации.

Типичная успешная сессия содержит ровно 8 событий: `thinking` → `tool_call(search_parts)` → `tool_result(search_parts)` → `tool_call(segment_object)` → `tool_result(segment_object)` → `tool_call(generate_image)` → `tool_result(generate_image)` → `result`. Это соответствует проектному описанию трёхшагового ReAct-пайплайна.

\ 

## 3.7. Выводы по главе

\ 

В ходе реализации системы были применены следующие ключевые технические решения:

::: {custom-style="MyNumberedList"}
Паттерн `lru_cache + run_in_executor` унифицирован для всех ML-моделей (rembg/U2Net, sentence-transformers, SAM 2, GroundingDINO). Это обеспечивает однократную инициализацию модели в памяти процесса и неблокирующее выполнение тяжёлых вычислений в асинхронном сервисе — event loop FastAPI остаётся доступным для обслуживания других запросов во время работы модели.
:::

::: {custom-style="MyNumberedList"}
Сегментационный пайплайн GroundingDINO → SAM 2 реализован с поддержкой множественных bounding box'ов и объединением частных масок через поэлементный максимум. Предзагрузка моделей при старте воркера в строго определённом порядке исключает конфликты инициализации `torch.jit`. Заглушка-синглтон позволяет разрабатывать и тестировать систему в целом без GPU-инфраструктуры.
:::

::: {custom-style="MyNumberedList"}
Системный промпт агента содержит конкретный запрет с примером, предотвращающий добавление языковой моделью визуальных деталей из обучающих данных. Это обеспечивает точное соответствие промпта генерации реальному изображению детали из каталога. Поддержка трёх LLM-провайдеров (OpenAI, Anthropic, OpenAI-совместимый) реализована через единую фабрику без изменения кода агента.
:::

::: {custom-style="MyNumberedList"}
Протокол `InpaintProvider` на базе Python `Protocol` позволяет подключать новые провайдеры генерации без наследования и без изменения кода агента. Специфика FLUX.2 Pro — явная нумерация ролей изображений в промпте и параллельная генерация вариантов через `asyncio.gather` с различными seed — обеспечивает максимальное качество и разнообразие результатов.
:::

::: {custom-style="MyNumberedList"}
SSE-стриминг реализован через опрос базы данных с интервалом 0.5 секунды, что не требует постоянного соединения между воркером и API-процессом. Такой подход упрощает горизонтальное масштабирование: несколько экземпляров API могут параллельно обслуживать стримы разных сессий, читая данные из общей БД.
:::

\ 

