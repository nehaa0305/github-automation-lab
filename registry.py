"""Model registry with versioning and model cards."""

import logging
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import semver

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Model registry with versioning and model cards."""
    
    def __init__(self, registry_path: Path = Path("models/registry")):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.registry_path / "models"
        self.cards_dir = self.registry_path / "cards"
        self.metrics_dir = self.registry_path / "metrics"
        
        for dir_path in [self.models_dir, self.cards_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def register_model(
        self,
        model_name: str,
        model_type: str,
        version: str,
        model_path: Path,
        metrics: Dict[str, float],
        training_data_info: Dict[str, Any],
        model_card: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model
            model_type: Type (linking, labeling, rag, risk)
            version: Semantic version
            model_path: Path to model files
            metrics: Performance metrics
            training_data_info: Information about training data
            model_card: Optional model card data
            
        Returns:
            Registered version string
        """
        # Validate version
        try:
            semver.VersionInfo.parse(version)
        except ValueError:
            raise ValueError(f"Invalid semantic version: {version}")
        
        # Create model directory
        model_dir = self.models_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        if model_path.is_file():
            shutil.copy2(model_path, model_dir / model_path.name)
        elif model_path.is_dir():
            shutil.copytree(model_path, model_dir, dirs_exist_ok=True)
        
        # Save metrics
        metrics_path = self.metrics_dir / f"{model_name}_{version}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create model card
        if model_card is None:
            model_card = self._create_default_model_card(
                model_name, model_type, version, metrics, training_data_info
            )
        
        card_path = self.cards_dir / f"{model_name}_{version}.md"
        with open(card_path, 'w', encoding='utf-8') as f:
            f.write(self._format_model_card(model_card))
        
        # Update registry index
        self._update_registry_index(model_name, model_type, version, model_dir)
        
        logger.info(f"Registered {model_name} v{version}")
        return version
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get latest version of a model."""
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            return None
        
        versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
        if not versions:
            return None
        
        # Sort by semantic version
        versions.sort(key=lambda v: semver.VersionInfo.parse(v), reverse=True)
        return versions[0]
    
    def get_model_path(self, model_name: str, version: Optional[str] = None) -> Optional[Path]:
        """Get path to model files."""
        if version is None:
            version = self.get_latest_version(model_name)
        
        if version is None:
            return None
        
        model_dir = self.models_dir / model_name / version
        return model_dir if model_dir.exists() else None
    
    def get_model_card(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get model card data."""
        if version is None:
            version = self.get_latest_version(model_name)
        
        if version is None:
            return None
        
        card_path = self.cards_dir / f"{model_name}_{version}.md"
        if not card_path.exists():
            return None
        
        return self._parse_model_card(card_path)
    
    def get_metrics(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, float]]:
        """Get model metrics."""
        if version is None:
            version = self.get_latest_version(model_name)
        
        if version is None:
            return None
        
        metrics_path = self.metrics_dir / f"{model_name}_{version}.json"
        if not metrics_path.exists():
            return None
        
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all registered models and versions."""
        models = {}
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
                versions.sort(key=lambda v: semver.VersionInfo.parse(v), reverse=True)
                models[model_dir.name] = versions
        
        return models
    
    def _create_default_model_card(
        self,
        model_name: str,
        model_type: str,
        version: str,
        metrics: Dict[str, float],
        training_data_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create default model card."""
        return {
            "model_name": model_name,
            "model_type": model_type,
            "version": version,
            "created_at": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "training_data": training_data_info,
            "description": f"{model_type.title()} model for GitHub automation",
            "use_cases": [
                "Automated PR/Issue generation",
                "Code quality assessment",
                "Risk scoring and merge decisions"
            ],
            "limitations": [
                "Performance may vary across different codebases",
                "Requires periodic retraining with new feedback"
            ],
            "bias_considerations": [
                "Trained on GitHub data which may have inherent biases",
                "Performance may vary across programming languages"
            ]
        }
    
    def _format_model_card(self, card_data: Dict[str, Any]) -> str:
        """Format model card as markdown."""
        lines = [
            f"# {card_data['model_name']} v{card_data['version']}",
            "",
            f"**Model Type**: {card_data['model_type']}",
            f"**Created**: {card_data['created_at']}",
            "",
            "## Description",
            card_data.get('description', ''),
            "",
            "## Metrics",
            ""
        ]
        
        # Add metrics table
        if 'metrics' in card_data:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for metric, value in card_data['metrics'].items():
                if isinstance(value, float):
                    lines.append(f"| {metric} | {value:.4f} |")
                else:
                    lines.append(f"| {metric} | {value} |")
            lines.append("")
        
        # Add training data info
        if 'training_data' in card_data:
            lines.extend([
                "## Training Data",
                "",
                f"- **Size**: {card_data['training_data'].get('size', 'Unknown')}",
                f"- **Time Range**: {card_data['training_data'].get('time_range', 'Unknown')}",
                f"- **Repositories**: {card_data['training_data'].get('repos', 'Unknown')}",
                ""
            ])
        
        # Add use cases
        if 'use_cases' in card_data:
            lines.extend([
                "## Use Cases",
                ""
            ])
            for use_case in card_data['use_cases']:
                lines.append(f"- {use_case}")
            lines.append("")
        
        # Add limitations
        if 'limitations' in card_data:
            lines.extend([
                "## Limitations",
                ""
            ])
            for limitation in card_data['limitations']:
                lines.append(f"- {limitation}")
            lines.append("")
        
        # Add bias considerations
        if 'bias_considerations' in card_data:
            lines.extend([
                "## Bias Considerations",
                ""
            ])
            for bias in card_data['bias_considerations']:
                lines.append(f"- {bias}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _parse_model_card(self, card_path: Path) -> Dict[str, Any]:
        """Parse model card from markdown."""
        # Simplified parser - would need more robust implementation
        with open(card_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract basic info
        lines = content.split('\n')
        model_name = lines[0].replace('# ', '').split(' v')[0]
        version = lines[0].split(' v')[1] if ' v' in lines[0] else '1.0.0'
        
        return {
            "model_name": model_name,
            "version": version,
            "content": content
        }
    
    def _update_registry_index(self, model_name: str, model_type: str, version: str, model_dir: Path):
        """Update registry index."""
        index_path = self.registry_path / "index.json"
        
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {}
        
        if model_name not in index:
            index[model_name] = {
                "model_type": model_type,
                "versions": [],
                "latest": version
            }
        
        if version not in index[model_name]["versions"]:
            index[model_name]["versions"].append(version)
            index[model_name]["versions"].sort(
                key=lambda v: semver.VersionInfo.parse(v), 
                reverse=True
            )
        
        index[model_name]["latest"] = index[model_name]["versions"][0]
        index[model_name]["model_type"] = model_type
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
