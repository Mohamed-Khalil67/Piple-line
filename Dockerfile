FROM python:3.6-slim

WORKDIR /usr/src/app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 88

CMD ["uvicorn", "main:app","--host", "0.0.0.0", "--port", "88"]