import launch

def update_transparent_background():
    from importlib.metadata import version as meta_version
    from packaging import version
    v = meta_version("transparent-background")
    print("current transparent-background " + v)
    if version.parse(v) < version.parse('1.2.3'):
        launch.run_pip("install -U transparent-background", "update transparent-background version for Ebsynth Utility")

if not launch.is_installed("transparent_background"):
    launch.run_pip("install transparent-background", "requirements for Ebsynth Utility")

update_transparent_background()

if not launch.is_installed("IPython"):
    launch.run_pip("install ipython", "requirements for Ebsynth Utility")

if not launch.is_installed("seaborn"):
    launch.run_pip("install ""seaborn>=0.11.0""", "requirements for Ebsynth Utility")

if not launch.is_installed("color_matcher"):
    launch.run_pip("install color-matcher", "requirements for Ebsynth Utility")

