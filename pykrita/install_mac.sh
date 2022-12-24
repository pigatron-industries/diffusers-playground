#!/bin/bash

cp stable_diffusion_krita.desktop ~/Library/Application\ Support/krita/pykrita
rsync -a --delete stable_diffusion_krita ~/Library/Application\ Support/krita/pykrita
