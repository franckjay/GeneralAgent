#This tutorial comes from https://docs.docker.com/get-started/part2/#share-your-image
#Another tutorial is https://www.civisanalytics.com/blog/using-docker-to-run-python/

#TO BUILD THIS THING:
#docker build -t friendlyhello2 .
#TO RUN THIS THING: docker run -v /Users/jfranck/Desktop/Dockerfiles/tutorial:/app friendlyhello2 python PlayingAgent.py
#OR: docker run -v /Users/jfranck/Desktop/Dockerfiles/tutorial:/app friendlyhello2 python CartPole.py

# Use an official Python runtime as a parent image
FROM python:2.7-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

RUN apt-get update && apt-get install -y \
 python-opengl \
 && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
#CMD ["python", "app.py"]
