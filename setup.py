import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="Unofficial FlashAttention2 with Custom Masks",
    version="0.1.0",
    packages=["fa2_custom_mask"],
    description='Unofficial implementation of FlashAttention2 with Custom Masks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "triton",
        "torch",
    ],
    python_requires=">=3.8",
)
