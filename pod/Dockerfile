FROM axolotlai/axolotl-cloud:main-latest

WORKDIR /workspace/finetuning

COPY builder/requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

RUN rm -rf /root/.cache/pip

COPY scripts .

RUN chmod +x autorun.sh
CMD ["/workspace/finetuning/autorun.sh"]
