from dataclasses import dataclass,field
from pathlib import Path
from typing import List, Dict, Optional
import random

@dataclass
class PromptConfigurator:
    """
    Genera prompt in base alla categoria di editing, mantenendo sempre l'oggetto.
    Le 'intensità' e i 'pool' sono personalizzabili.
    """
    # Pools per categorie
    materials: List[str] = field(default_factory=lambda: [
        "polished steel", "brushed aluminum", "matte black metal", "transparent acrylic",
        "solid oak wood", "walnut wood", "marble stone", "ceramic", "carbon fiber"
    ])
    colors: List[str] = field(default_factory=lambda: [
        "white", "black", "blue", "green", "red",
        "pink", "yellow", "purple", "gray"
    ])
    styles: List[str] = field(default_factory=lambda: [
        "minimalist", "industrial", "futuristic", "baroque",
        "mid-century modern", "Japanese wabi-sabi", "Art Deco"
    ])
    add_components: List[str] = field(default_factory=lambda: [
        "with brass handles", "with a small side shelf", "with a leather cushion",
        "with a frosted glass cover", "with a chrome footrest", "with a cable management clip"
    ])

    # Contesto descrittivo fisso per “qualità” (puoi variarlo)
    suffix: str = "without changing background, high detail, photorealistic"

    #Sceglie (in base al seed) n elementi dal pool di cambiamenti disponibili per la categoria selezionata
    def _pick(self, pool: List[str], n: int, seed: Optional[int]) -> List[str]:
        rng = random.Random(seed)
        return [rng.choice(pool) for _ in range(n)]

    def make_prompts(
        self,
        object_name: str,
        category: str,
        n: int,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        category: one of {"change_material", "change_color", "change_style", "add_component"}
        strength: opzionale ("subtle" | "moderate" | "strong") che modula testo
        """

        if category == "change_material":
            choices = self._pick(self.materials, n, seed)
            return [f"a photo of a {object_name} made of {c} {self.suffix}".strip() for c in choices]

        if category == "change_color":
            choices = self._pick(self.colors, n, seed)
            return [f"a photo of a {object_name} in {c} color {self.suffix}".strip() for c in choices]

        if category == "change_style":
            choices = self._pick(self.styles, n, seed)
            return [f"a photo of a {object_name} in {c} style {self.suffix}".strip() for c in choices]

        if category == "add_component":
            choices = self._pick(self.add_components, n, seed)
            return [f"a photo of a {object_name} {c} {self.suffix}".strip() for c in choices]

        raise ValueError(f"Unknown category: {category}")


promptConfig = PromptConfigurator()
print(promptConfig.make_prompts("cup","add_component", 3, seed=1))