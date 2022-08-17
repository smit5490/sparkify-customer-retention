# Dockerfile used to create an image to deploy on Heroku
# Base image will the official python 3.8 version
FROM python:3.8

# Move files to app folder
COPY . /app

# Set working directory
WORKDIR /app

# Install python dependencies
RUN pip3 install -r docker_requirements.txt

# Expose port 8501
EXPOSE 8501

# Run streamlit command and pass PORT so Heroku can correctly render app
CMD streamlit run --server.port $PORT app.py

