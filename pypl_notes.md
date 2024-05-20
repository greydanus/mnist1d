# SJG Notes for PYPL uploads:


* `git tag v0.0.2`
* `git push origin v0.0.2`
* (commit)
* `python3 -m build`
* Upload w/`twine` (Test)
  * `python3 -m twine upload --repository testpypi dist/*`
  * `python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps mnist1d`
  * View on `https://test.pypi.org/project/mnist1d/`
* Upload w/`twine` (Main)
  * `python3 -m twine upload dist/*`
  * `python3 -m pip install mnist1d`
  * View on `https://pypi.org/project/mnist1d/`