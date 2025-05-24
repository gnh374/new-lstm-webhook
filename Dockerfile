# Gunakan base image dengan TensorFlow
FROM python:3.9-slim

# Set working directory di container
WORKDIR /app

# Salin file proyek ke dalam container
COPY . .

RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    openssh-client \
    && curl -fsSL https://releases.hashicorp.com/terraform/1.5.5/terraform_1.5.5_linux_amd64.zip -o terraform.zip \
    && unzip terraform.zip \
    && mv terraform /usr/local/bin/ \
    && rm terraform.zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ubah permission file terraform.pem agar bisa digunakan SSH
RUN chmod 600 terraform.pem

# Install dependency Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port (sesuaikan jika pakai Flask/FastAPI)
EXPOSE 5000

# Jalankan aplikasi (ganti dengan perintah yang sesuai)
CMD ["python", "app.py"]
