#!/bin/bash
mkdocs build
rsync -av site/ public/
