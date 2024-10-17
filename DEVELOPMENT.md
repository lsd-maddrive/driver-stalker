# Development guide

This is guide how to prepare development environment and use main tools

## Table of contents

- [Table of contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Initialize your code](#initialize-your-code)
- [Optional setup steps](#optional-setup-steps)

## Prerequisites

> If you need to install tools, check links

- [Python](docs/TOOLS.md#python)
- [Poetry](docs/TOOLS.md#poetry)
- [Make](docs/TOOLS.md#make)

## Initialize your code

1. Initialize poetry and install `pre-commit` hooks:

```bash
make project-init
```

[Table of contents](#table-of-contents)

## Optional setup steps

1. Install VSCode Extensions
   - [EditorConfig for VS Code](https://marketplace.visualstudio.com/items?itemName=EditorConfig.EditorConfig)
      Open control panel (Ctrl+P) and enter `ext install EditorConfig.EditorConfig`

1. To initialize generation of Table of Contents from notebook headers we use `nbextension`:

    - `nbextention-toc-install`

    > It is required version nbconvert~=5.6.1 (checked for date 2021-12-29)

    - To export notebook with ToC use next command:

      ```bash
      poetry run jupyter nbconvert --template toc2 --to html_toc --output-dir ./exports <путь до файла>
      ```

      > For example, `poetry run jupyter nbconvert --template toc2 --to html_toc --output-dir ./exports notebooks/example.ipynb`

      To use embedded images into HTML use option `html_embed`:

      ```bash
      poetry run jupyter nbconvert --template toc2 --to html_embed --output-dir ./exports <путь до файла>
      ```

[Table of contents](#table-of-contents)
