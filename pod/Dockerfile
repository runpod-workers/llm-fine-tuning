FROM axolotlai/axolotl-cloud:main-latest

WORKDIR /workspace/fine-tuning

COPY builder/requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

RUN rm -rf /root/.cache/pip

COPY scripts/WELCOME /etc/motd

COPY scripts .

RUN chmod +x autorun.sh
CMD ["/workspace/fine-tuning/autorun.sh"]
