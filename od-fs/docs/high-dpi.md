

# macOS

On macOS, it seems that on high-DPI displays, you often end up getting an internal rendering target
that is 2x the logical resolution, which macOS then downscales when outputting to the screen.

The end result (if true - may need more investigation) is that in order to get pixel perfect
rendering, you may have to ensure that the macOS logical resolution is either the same as the
real pixel resolution of the display - or half. So on a 2560x1600 display, you should choose the
2560x1600 resolution or 1280x800 in display settings to get accurate results and avoid an extra
scaling step.
