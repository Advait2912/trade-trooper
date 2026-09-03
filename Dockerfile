FROM python:3.12-slim

WORKDIR /app

# Install system dependencies and official Alpaca CLI binary
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -sL https://github.com/alpacahq/cli/releases/download/v0.0.14/cli_0.0.14_linux_amd64.tar.gz | tar -xz -C /usr/local/bin \
    && chmod +x /usr/local/bin/alpaca \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run unified portfolio trader across the 11 tickers
CMD ["python", "main.py", "--universe", "NVDA,AAPL,MSFT,AMD,JPM,BAC,V,GS,TSLA,XOM,KO", "--trade"]
