# Sandbox for the segmentation tools: Python + imaging libraries, nothing else.
# Run with --network none; only /data (read-only) and /out are mounted.
FROM python:3.12-slim
RUN pip install --no-cache-dir \
        numpy==2.3.1 scipy==1.16.0 nibabel==5.4.0 matplotlib==3.10.3 mcp==1.27.2 \
    && useradd --uid 1000 --create-home agent
COPY tools.py /tools/tools.py
ENV MPLCONFIGDIR=/tmp PYTHONUNBUFFERED=1
USER agent
CMD ["python", "/tools/tools.py"]
