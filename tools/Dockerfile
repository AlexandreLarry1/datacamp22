FROM codalab/codalab-legacy:py39

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Step 2: Install system-level dependencies (if any)
# e.g., git, wget, or common libraries for OpenCV like libgl1
RUN pip install -U pip

# Step 3: Copy and pre-install all Python dependencies
# This 'requirements.txt' file should list pandas, scikit-learn, timm, etc.
# Place it in the same directory as this Dockerfile.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
