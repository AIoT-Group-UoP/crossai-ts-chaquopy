from setuptools import setup

setup(
    name="crossai-ts",
    version="0.0.0.1",
    packages=[
        "caits",
        "caits.base",
        "caits.dataset",
        "caits.fe",
        "caits.fe.core_spectrum",
        "caits.transformers",
        "caits.performance"
    ],
    url="https://github.com/AIoT-Group-UoP/crossai-ts-chaquopy",
    license="Apache License 2.0",
    author="Pantelis Tzamalis, George Kontogiannis",
    author_email="tzamalis@ceid.upatras.gr",
    description="An open-source Python library for developing "
                "end-to-end AI pipelines for Time Series Analysis on "
                "Android devices using the Chaquopy SDK.",
    install_requires=[
        "pandas==2.2.0",
        "scipy==1.12.0",
        "scikit-learn==1.4.0",
        "resampy==0.4.2",
        "samplerate==0.1.0; platform_system=='Linux'",
        "soxr==0.3.7",
    ],
    python_requires=">=3.8"
)
