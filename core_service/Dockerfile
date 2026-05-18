FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

WORKDIR /app

COPY pyproject.toml README.md .
RUN uv pip install --system -e .

# SAM2 and GroundingDINO must be installed manually from source
# RUN pip install git+https://github.com/facebookresearch/sam2.git
# RUN pip install groundingdino-py

COPY . .

RUN mkdir -p /app/storage/originals /app/storage/masks /app/storage/results

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
