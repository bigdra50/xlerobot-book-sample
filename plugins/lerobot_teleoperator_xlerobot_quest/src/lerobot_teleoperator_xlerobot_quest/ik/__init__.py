"""VR teleop の IK。

- so101_dual、dual_urdf: ik_backend=pyroki の両腕 URDF。jax / pyroki / yourdfpy を読むのは so101_dual だけ
- analytic、hand_axes、vendor: closed_form。URDF も jax も読まない
"""
