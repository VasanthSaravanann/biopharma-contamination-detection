FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# Install requirements first
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Install CPU PyTorch wheel explicitly for reproducibility
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.0 torchvision==0.15.2

COPY . /app

ENV PYTHONPATH=/app

CMD ["pytest", "-q"]
