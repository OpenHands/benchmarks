# Juan's development VM
resource "google_compute_instance" "juan-dev" {
  name         = "juan-dev"
  machine_type = "n1-standard-4"
  zone         = "us-central1-a"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata = {
    owner = "juan"
    purpose = "development"
  }
}
