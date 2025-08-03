# Use an official Python runtime as a parent image
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .

# Install dependencies using uv
RUN pip install uv && \
    uv pip install --system .

COPY . .

# Build the model so it's included in the image
RUN python build.py

# Expose the port Streamlit runs on
EXPOSE 8501

# Define the command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]