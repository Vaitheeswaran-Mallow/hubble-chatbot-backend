FROM python:3.11-slim

RUN pip install --upgrade pip && pip install uv

WORKDIR /app

COPY . /app

RUN rm -rf .venv
RUN uv sync

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

EXPOSE 8000

CMD ["uv", "run", "python", "main.py"]
