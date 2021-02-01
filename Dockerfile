#ENV APP_HOME /app 
#WORKDIR ${APP_HOME}

#COPY . ./

#RUN pip install pip pipenv --upgrade
#RUN python -m pip install --upgrade pip
#RUN pip install streamlit

#COPY requirements.txt .
#RUN python -m pip install -r requirements.txt

#RUN ["chmod", "+x", "./scripts/entrypoint.sh"]

#EXPOSE 8000
#CMD ["./scripts/entrypoint.sh"]

# Version Python alpine est plus petit et pyhton 3.6 
FROM python:3.8-slim 
COPY . /app
WORKDIR /app
RUN python -m pip install --upgrade pip
RUN pip install streamlit
RUN pip install -r requirements.txt
EXPOSE 8000
ENTRYPOINT ["streamlit","run"]
CMD ["main.py"]
