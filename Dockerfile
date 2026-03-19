# base image
FROM python:3.13.9

#workdir
WORKDIR /app

# copy
COPY . /app

# run
RUN pip install -r requirements.txt

# port 
EXPOSE 5000

#command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]