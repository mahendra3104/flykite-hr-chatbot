# Use a minimal base image with Python 3.9 installed
FROM python:3.9

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory inside the container for the application
WORKDIR $HOME/app

# Copy application files from the repository root to the container's /home/user/app
COPY --chown=user src/ ./src/

# Install llama-cpp-python specifically for CPU (assuming a CPU-only Space)
# This is done separately to ensure it's built correctly without CUDA dependencies if not available.
RUN pip install llama-cpp-python==0.1.85 --force-reinstall --no-cache-dir CMAKE_ARGS="-DLLAMA_CUBLAS=off"

# Copy and install other Python dependencies listed in requirements.txt
COPY --chown=user requirements.txt .
RUN pip3 install -r requirements.txt

# Define the command to run the Streamlit app on port "8501" and make it accessible externally
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
