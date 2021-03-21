FROM python:3.7.3-slim
LABEL maintainer="Beth Baumann <baumann.bethany@gmail.com>"
LABEL version="0.1"
LABEL description="ClinVar and CIViC abstract scoring app"

WORKDIR /data

COPY . /data

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip && pip install lxml && pip install -r requirements.txt

EXPOSE 8501

CMD streamlit run ./data/clinvar_streamlit.py


