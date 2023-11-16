# Use an official Python runtime as a parent image
FROM python:3.12-slim
# Install python3.x-dev package
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
