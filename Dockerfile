#dockerfile, image, container
FROM python:3.11-slim

# environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

#system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

#install uv
RUN pip install --upgrade pip uv

#working directory
WORKDIR /app

#copy dependencies files first 
COPY pyproject.toml uv.lock ./

#install dependencies
RUN uv pip install --system --no-cache .

#copy project code
COPY . .

#expose streamlit port
EXPOSE 8501

#run streamlit app
CMD ["streamlit","run","medibot.py","--server.port=8501","--server.address=0.0.0.0"]

