terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Configure the AWS Provider
provider "aws" {
  region = "us-east-1"
  assume_role {
    role_arn = "arn:aws:iam::709412559461:role/LabRole"
  }
}

data "aws_vpc" "existing_vpc" {
  id = "vpc-0e3adfd890502179b"
}

# Get all AMIs with the Cluster-1 tag
data "aws_ami_ids" "cluster_amis" {
  owners = ["self"]
  
  filter {
    name   = "tag:Cluster"
    values = ["Cluster-1"]
  }
}

# Get detailed information for each AMI
data "aws_ami" "detailed_amis" {
  for_each = toset(data.aws_ami_ids.cluster_amis.ids)
  
  owners = ["self"]
  most_recent = true
  
  filter {
    name   = "image-id"
    values = [each.value]
  }
}

data "aws_security_group" "existing" {
  id = "sg-063ebc228f5d68027"  # Replace with your security group ID
 
}

data "aws_iam_instance_profile" "existing_profile" {
  name = "LabInstanceProfile"  
}




# Create an instance for each AMI
resource "aws_instance" "cluster_instances" {
  for_each = data.aws_ami.detailed_amis
  
  ami           = each.value.id
  instance_type = lookup(each.value.tags, "InstanceType", "t3.medium")
  vpc_security_group_ids = [data.aws_security_group.existing.id]
  iam_instance_profile   = data.aws_iam_instance_profile.existing_profile.name
  
  tags = {
    Name = "Restored-Instance-${each.key}"
    AMI_ID = each.value.id
    Original_AMI_Name = each.value.name
  }
}

# Output all created instance IDs
output "instance_ids" {
  value = {for k, v in aws_instance.cluster_instances: k => v.id}
}

# Output all AMI IDs that were used
output "ami_ids" {
  value = data.aws_ami_ids.cluster_amis.ids
}