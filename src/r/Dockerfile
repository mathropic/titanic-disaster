FROM rocker/r-ver:4.6.1

WORKDIR /app/r

COPY install_packages.R .
RUN Rscript install_packages.R

COPY main.R .

CMD ["Rscript", "main.R"]