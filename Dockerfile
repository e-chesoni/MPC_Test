FROM ubuntu:latest
LABEL authors="elainechesoni"

ENTRYPOINT ["top", "-b"]

# Use an official Python runtime as a parent image
FROM python:3.11.6

# Set the working directory in the container
WORKDIR .

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run app.py when the container launches
CMD ["python", "main.py"]
