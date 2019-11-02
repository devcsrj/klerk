# Klerk

![](https://img.shields.io/github/license/devcsrj/klerk)
![](https://img.shields.io/travis/devcsrj/klerk)

> This project is still under construction. 🚧

## Prerequisites

* JDK 8


## Pipelines

All pipelines are configured as tasks in gradle.

To download journals:
```shell script
./gradlew collateJournals --args='--input=17 --output=dist'`
```

...where `input` is a comma-separated value of which Congress 
to include, and `output` is the output directory.

To extract the text from journals:
```shell script
./gradlew extractJournals --args='--dir=dist'
```

...where `dir` is directory where the downloaded journals reside.