terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5.1"
    }
  }
}

# Configure the AWS Provider
provider "aws" {
  region = "us-east-1"
  assume_role {
    role_arn = "arn:aws:iam::872717238482:role/LabRole"
  }
}

# Generate a random token for K3s
resource "random_password" "k3s_token" {
  length  = 32
  special = false
}

# Fetch the existing VPC
data "aws_vpc" "existing_vpc" {
  id = "vpc-09bad494b664adc15"
}

# Fetch the existing security group
data "aws_security_group" "existing" {
  id = "sg-0011cfb37cfee265b"
}

# Fetch the existing IAM instance profile
data "aws_iam_instance_profile" "existing_profile" {
  name = "LabInstanceProfile"
}

# Find the latest Ubuntu 22.04 AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Create Master Node
resource "aws_instance" "master_node" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t2.medium"
  
  vpc_security_group_ids = [data.aws_security_group.existing.id]
  iam_instance_profile   = data.aws_iam_instance_profile.existing_profile.name

  # Ensure SSH access
  key_name = "terraform"  # Assuming you're using the default lab key

  tags = {
    Name = "Master-Node"
    Role = "Cluster-Master"
  }
}

# Create Worker Node
resource "aws_instance" "worker_node" {

  depends_on = [aws_instance.master_node]
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t2.small"
  
  vpc_security_group_ids = [data.aws_security_group.existing.id]
  iam_instance_profile   = data.aws_iam_instance_profile.existing_profile.name

  # Ensure SSH access
  key_name = "terraform"  # Assuming you're using the default lab key

  tags = {
    Name = "Worker-Node"
    Role = "Cluster-Worker"
  }
}


# Install K3s on Master Node
resource "null_resource" "k3s_master_setup" {
  # Trigger on master node creation
  triggers = {
    master_instance_id = aws_instance.master_node.id
  }

  # SSH connection details
  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("./terraform.pem")  # Path to your SSH key
    host        = aws_instance.master_node.public_ip
  }

  # Provision K3s master
  provisioner "remote-exec" {
    inline = [
      "sudo apt-get update",
      "sudo apt-get install -y curl",
      "curl -sfL https://get.k3s.io | K3S_TOKEN=${random_password.k3s_token.result} sh -s - server",
      "sudo chmod 644 /etc/rancher/k3s/k3s.yaml",
    ]
  }
}

resource "null_resource" "copy_kubeconfig" {
  depends_on = [null_resource.k3s_master_setup]

  provisioner "local-exec" {
    command = <<EOT
      scp -o StrictHostKeyChecking=no -i ./terraform.pem ubuntu@${aws_instance.master_node.public_ip}:/etc/rancher/k3s/k3s.yaml ./k3s.yaml
      chmod 644 ./k3s.yaml
    EOT
  }
}




# Install K3s Agent on Worker Node
resource "null_resource" "k3s_worker_setup" {
  # Trigger on worker node creation and after master setup
  triggers = {
    worker_instance_id = aws_instance.worker_node.id
    master_instance_id = aws_instance.master_node.id
  }

  # Depends on master node setup
  depends_on = [null_resource.copy_kubeconfig ]
  
  # SSH connection details
  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("./terraform.pem")  # Path to your SSH key
    host        = aws_instance.worker_node.public_ip
  }

  
  provisioner "file" {
    source      = "./k3s.yaml" # Dari lokal Terraform
    destination = "/home/ubuntu/k3s.yaml"

  }


  # Provision K3s worker
  provisioner "remote-exec" {
    inline = [
      "sudo apt-get update",
      "sudo apt-get install -y curl",
      "curl -sfL https://get.k3s.io | K3S_TOKEN=${random_password.k3s_token.result} K3S_URL=https://${aws_instance.master_node.private_ip}:6443 sh -s - agent",
       "sed -i 's/127.0.0.1/${aws_instance.master_node.private_ip}/g' /home/ubuntu/k3s.yaml",
      "mkdir -p ~/.kube",
      "cp /home/ubuntu/k3s.yaml ~/.kube/config",
      "chmod 600 ~/.kube/config",
      "export KUBECONFIG=~/.kube/config",
      "echo 'export KUBECONFIG=~/.kube/config' >> ~/.bashrc",
      "sudo systemctl restart k3s-agent"
    ]
  }

 
}


resource "null_resource" "rancher_cluster_registration" {
  # Trigger on master node creation and after K3s setup
  triggers = {
    master_instance_id = aws_instance.master_node.id
  }

  # Depends on K3s master setup
  depends_on = [null_resource.k3s_master_setup]

  # SSH connection details
  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("./terraform.pem")
    host        = aws_instance.master_node.public_ip
  }

  # Provision Rancher registration
  provisioner "remote-exec" {
    inline = [
      # Install necessary tools
      "sudo apt-get update",
      "sudo apt-get install -y curl wget jq",

      # Download and install Rancher CLI
      "wget https://github.com/rancher/cli/releases/download/v2.7.0/rancher-linux-amd64-v2.7.0.tar.gz",
      "tar -xzvf rancher-linux-amd64-v2.7.0.tar.gz",
      "sudo mv rancher-v2.7.0/rancher /usr/local/bin/rancher",
      "rm -rf rancher-linux-amd64-v2.7.0.tar.gz rancher-v2.7.0",

      # Create the Rancher registration script
      <<-EOF
      cat << 'EOL' > /home/ubuntu/rancher_register.sh
      #!/bin/bash
      set -e
      set -x

      export RANCHER_URL="https://3.208.173.140.sslip.io"
      export RANCHER_ACCESS_KEY="token-rtv46"
      export RANCHER_SECRET_KEY="mdv65cpslx4r2b2rcqnkkrnwgq9brmfhlrb2l694jrds46h2fpg5tg"
      export CLUSTER_NAME="backup-cluster"
      export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

      # Create cluster and get Cluster ID
       CLUSTER_ID=$(curl -k -X POST -H "Authorization: Bearer $RANCHER_ACCESS_KEY:$RANCHER_SECRET_KEY" \
      -H "Content-Type: application/json" \
      "$RANCHER_URL/v3/clusters" \
      -d '{
        "type": "cluster",
        "name": "'"$CLUSTER_NAME"'",
        "k3sConfig": {
          "kubernetesVersion": "v1.27.4+k3s1"
        }
      }' | jq -r '.id')

      if [[ -z "$CLUSTER_ID" ]]; then
        echo "Failed to get Cluster ID!"
        exit 1
      fi

      echo "Cluster ID: $CLUSTER_ID"
      sleep 1
      TOKEN_RESPONSE=$(curl -k -X POST \
      -H "Authorization: Bearer $RANCHER_ACCESS_KEY:$RANCHER_SECRET_KEY" \
      -H "Content-Type: application/json" \
      "$RANCHER_URL/v3/clusterRegistrationTokens" \
      -d '{
          "type": "clusterRegistrationToken",
          "clusterId": "'"$CLUSTER_ID"'"
      }')

      # Extract the token
      REGISTRATION_TOKEN=$(echo "$TOKEN_RESPONSE" | jq -r '.token // empty')


      if [[ -z "$REGISTRATION_TOKEN" ]]; then
          echo "Failed to generate registration token!"
          exit 1
      fi

      curl --insecure -sfL "https://3.208.173.140.sslip.io/v3/import/$${REGISTRATION_TOKEN}_$${CLUSTER_ID}.yaml"  | kubectl apply -f -
      
      EOL
      EOF
      ,

      # Set executable permission and execute the script
      "chmod +x /home/ubuntu/rancher_register.sh",
      "bash /home/ubuntu/rancher_register.sh"
    ]
  }
}

resource "null_resource" "install_velero" {
  depends_on = [null_resource.k3s_master_setup]

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("./terraform.pem")
    host        = aws_instance.master_node.public_ip
  }

  provisioner "remote-exec" {
    inline = [
      "export KUBECONFIG=/etc/rancher/k3s/k3s.yaml",
      "export VELERO_VERSION=$(curl -s https://api.github.com/repos/vmware-tanzu/velero/releases/latest | grep tag_name | cut -d '\"' -f 4)",
      "curl -LO https://github.com/vmware-tanzu/velero/releases/download/$${VELERO_VERSION}/velero-$${VELERO_VERSION}-linux-amd64.tar.gz",
      "tar -xvf velero-$${VELERO_VERSION}-linux-amd64.tar.gz",
      "sudo mv velero-$${VELERO_VERSION}-linux-amd64/velero /usr/local/bin/",
      "rm -rf velero-$${VELERO_VERSION}-linux-amd64*",
      "kubectl create namespace velero || true",
      "velero install --provider aws --plugins velero/velero-plugin-for-aws:v1.10.0 --bucket new-velero-naomi --prefix new-velero-naomi --backup-location-config region=us-east-1,s3Url=https://s3.us-east-1.amazonaws.com --snapshot-location-config region=us-east-1 --use-node-agent --no-secret --wait",
      "kubectl get pods -n velero",
      "velero restore create --from-backup backup-nginx",
      "kubectl get pods -n nginx"
    ]
  }
}




# Outputs
output "master_node_id" {
  value = aws_instance.master_node.id
}

output "worker_node_id" {
  value = aws_instance.worker_node.id
}

output "master_node_public_ip" {
  value = aws_instance.master_node.public_ip
}

output "worker_node_public_ip" {
  value = aws_instance.worker_node.public_ip
}

output "k3s_token" {
  value     = random_password.k3s_token.result
  sensitive = true
}