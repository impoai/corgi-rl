# Corgi concept render

![Corgi robot concept render](assets/corgi-concept.jpg)

This image illustrates a proposed finished appearance. It is not a photograph,
a manufacturing design, or a render of the current MJCF. Joint packaging, shell
clearances, mass distribution, and hardware feasibility still need engineering.

## Generation details

- Provider: Google Gemini API, called directly with the owner's existing API key.
- Model: `gemini-3-pro-image`.
- Generated: October 27, 2026.
- Requested format: 16:9, 2K. Returned image: JPEG, 2752 × 1536 pixels.
- Repository asset: `docs/assets/corgi-concept.jpg`.
- SHA-256: `1095613a329a1326b99452b9e86ea09a2c324e56d00bde5921d3252aa429b7db`.
- The provider's returned image is stored without visual edits.
- Credentials and request scripts are not included in this repository.

API configuration follows the [official Gemini image-generation documentation](https://ai.google.dev/gemini-api/docs/generate-content/image-generation).

## Generation prompt

```text
Use case: product-mockup.
Asset: a wide, polished hero image for the Corgi RL open-source robotics repository.
Create one photorealistic industrial-design concept render of a compact, low-cost, corgi-inspired quadruped robot. It is a proposed finished product, not an animal and not a plush toy. Give it a long, low, softly rounded rectangular torso, short mechanically credible legs, a small friendly fixed head with two upright triangular corgi ears, and a small black sensor face. Keep the overall design simple enough to imagine 3D-printing and assembling with accessible smart servos.
The robot has exactly four legs, with three actuated joints per leg: a hip abduction joint at the torso, a hip pitch joint, and a knee pitch joint. Show understandable cylindrical servo housings, clean charcoal metal linkages, discreet fasteners, and four rounded black rubber feet. The head and ears are fixed shell features, not additional motorized joints. All four legs must be distinct and grounded in a relaxed, stable standing pose.
Materials: warm tan-orange matte polymer upper shell, cream-white chest and lower shell, dark graphite joint modules, satin aluminum details. Precise panel seams, a subtle cooling vent and serviceable construction. Friendly and capable, restrained and contemporary, with no ornamental sci-fi armor or excessive exposed wires.
Composition: one robot only, front three-quarter view at robot eye level, showing the face, left side and the four-legged construction. Entire robot visible with generous breathing room. It should occupy about two-thirds of the frame width. Wide 16:9 landscape image. Quiet warm off-white seamless studio floor and backdrop, soft contact shadow, softbox lighting, crisp but natural material detail, premium product visualization.
No text, letters, logos, watermarks, captions, diagram labels, people, extra robots, wheels, extra limbs, fur or biological anatomy. Do not imitate an existing branded robot. The visual must read clearly as an original corgi-shaped robot concept.
```
