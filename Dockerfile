FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run unified portfolio trader across the 11 tickers
CMD ["python", "main.py", "--universe", "NVDA,AAPL,MSFT,AMD,JPM,BAC,V,GS,TSLA,XOM,KO", "--trade"]
