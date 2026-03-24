#!/bin/bash
# Intel GPU Sürücüleri (Level Zero, OpenCL) Kurulum Betiği
# This script is required for the Intel Arc GPU to be recognized by PyTorch (IPEX) on Linux.

echo "Intel GPU Repo Anahtarları ekleniyor..."
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --dearmor --yes --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

echo "Paketler güncelleniyor ve Intel Sürücüleri kuruluyor..."
sudo apt-get update
sudo apt-get install -y \
  intel-opencl-icd intel-level-zero-gpu level-zero \
  intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
  libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
  libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
  mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all \
  linux-headers-$(uname -r) \
  execstack

echo "IPEX (Intel Extension for PyTorch) 'executable stack' izin sorunu çözülüyor..."
for lib in /home/tuncay/PycharmProjects/CodeDuplicationDetection/.venv/lib/python3.11/site-packages/intel_extension_for_pytorch/*.so; do
    if [ -f "$lib" ]; then
        sudo execstack -c "$lib"
    fi
done

echo "Kurulum tamamlandı! Değişikliklerin aktif olması için bilgisayarınızı yeniden başlatmanız (veya yeniden oturum açmanız) önerilir."
echo "PyTorch ile kontrol etmek için: python -c 'import torch; import intel_extension_for_pytorch; print(torch.xpu.is_available())'"
