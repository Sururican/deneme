# Use an official Python runtime as a parent image
FROM python:3.9-slim
RUN apt-get update && apt-get install -y python3.12-dev
# Install python3.9-dev package
RUN apt-get update && apt-get install -y python3-distutils
# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run lstm.py when the container launches
CMD ["python", "LSTM-2.py"]