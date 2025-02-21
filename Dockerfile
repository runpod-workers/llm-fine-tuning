FROM axolotlai/axolotl-cloud:main-latest

WORKDIR /root/finetuning

COPY builder/requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

RUN rm -rf /root/.cache/pip

COPY src .

RUN chmod +x run.sh
CMD ["/root/finetuning/run.sh"]
