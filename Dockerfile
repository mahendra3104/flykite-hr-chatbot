# Use a minimal base image with Python 3.9 installed
FROM python:3.9

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory inside the container for the application
WORKDIR $HOME/app

# Copy application files and data from the repository root to the container's /home/user/app
# Assuming the build context (Hugging Face Space root) contains:
# src/streamlit_app.py, requirements.txt, data/Dataset - Flykite Airlines_ HRP.pdf
COPY --chown=user src/ ./src/
COPY --chown=user requirements.txt .
COPY --chown=user data/ ./data/

# Install Python dependencies listed in requirements.txt
RUN pip3 install -r requirements.txt

# Define the command to run the Streamlit app on port "8501" and make it accessible externally
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
