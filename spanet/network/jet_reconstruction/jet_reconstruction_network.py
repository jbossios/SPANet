# python imports
import numpy as np
from sklearn import metrics as sk_metrics
from typing import Tuple, Dict, Callable
from collections import OrderedDict

# torch imports
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import pytorch_lightning as pl

# spanet imports
from spanet.network.prediction_selection import extract_predictions
from spanet.network.layers.branch_decoder import BranchDecoder
from spanet.network.layers.jet_encoder import JetEncoder
from spanet.options import Options
from spanet.dataset.event_info import EventInfo
from spanet.network.utilities.divergence_losses import jet_cross_entropy_loss
from spanet.dataset.evaluator import SymmetricEvaluator
from spanet.network.learning_rate_schedules import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

class JetReconstructionNetwork(pl.LightningModule):
    def __init__(self, options: Options):
        """ Base class defining the SPANet architecture.

        Parameters
        ----------
        options: Options
            Global options for the entire network.
            See network.options.Options
        """

        super().__init__()

        self.save_hyperparameters(options)

        self.options = options
        event_info = EventInfo.read_from_ini(options.event_info_file)
        self.num_children = len(event_info.targets[event_info.event_particles[0]][0]) # NOTE: for more complicated decays you may need to update this line and propagate the changes. This current assumes symmetric pair production.
        self.target_symmetries = event_info.mapped_targets.items()
        self.num_features = event_info.num_features
        self.hidden_dim = options.hidden_dim
        self.enable_softmax = True

        # Shared options for all transformer layers
        transformer_options = (options.hidden_dim,
                               options.num_attention_heads,
                               options.hidden_dim,
                               options.dropout,
                               options.transformer_activation)

        self.encoder = JetEncoder(options, self.num_features, transformer_options)
        self.decoders = nn.ModuleList([
            BranchDecoder(options, size, permutation_indices, transformer_options, self.enable_softmax)
            for _, (size, permutation_indices) in self.target_symmetries
        ])

        event_permutation_group = np.array(event_info.event_permutation_group)
        self.event_permutation_tensor = torch.nn.Parameter(torch.from_numpy(event_permutation_group), False)

        self.evaluator = SymmetricEvaluator(event_info)


    def forward(self, source_data: Tensor, source_mask: Tensor) -> Tuple[Tuple[Tensor, Tensor], ...]:

        # Extract features from data using transformer
        hidden, padding_mask, sequence_mask = self.encoder(source_data, source_mask)

        # Pass the shared hidden state to every decoder branch
        return tuple(decoder(hidden, padding_mask, sequence_mask) for decoder in self.decoders)

    def predict_jets(self, source_data: Tensor, source_mask: Tensor) -> np.ndarray:
        # Run the base prediction step
        with torch.no_grad():
            predictions = []
            for prediction, _, _ in self.forward(source_data, source_mask):
                prediction[torch.isnan(prediction)] = -np.inf
                predictions.append(prediction)

            # Find the optimal selection of jets from the output distributions.
            return extract_predictions(predictions)

    def predict_jets_and_particle_scores(self, source_data: Tensor, source_mask: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            predictions = []
            scores = []
            for prediction, classification, _ in self.forward(source_data, source_mask):
                prediction[torch.isnan(prediction)] = -np.inf
                predictions.append(prediction)

                scores.append(torch.sigmoid(classification).cpu().numpy())
            scores = np.stack(scores)
            # protect against batch size of 1
            if len(scores.shape) == 1:
                scores = np.expand_dims(scores,-1)
            return extract_predictions(predictions), scores

    def predict_jets_and_particles(self, source_data: Tensor, source_mask: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        predictions, scores = self.predict_jets_and_particle_scores(source_data, source_mask)

        # Always predict the particle exists if we didn't train on it
        if self.options.classification_loss_scale == 0:
            scores += 1

        return predictions, scores >= 0.5

    def particle_classification_loss(self, classification: Tensor, target_mask: Tensor) -> Tensor:
        loss = F.binary_cross_entropy_with_logits(classification, target_mask.float(), reduction='none')
        return self.options.classification_loss_scale * loss

    def negative_log_likelihood(self,
                                predictions: Tuple[Tensor, ...],
                                classifications: Tuple[Tensor, ...],
                                targets: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
        # We are only going to look at a single prediction points on the distribution for more stable loss calculation
        # We multiply the softmax values by the size of the permutation group to make every target the same
        # regardless of the number of sub-jets in each target particle
        predictions = [prediction + torch.log(torch.scalar_tensor(decoder.num_targets))
                       for prediction, decoder in zip(predictions, self.decoders)]

        # Convert the targets into a numpy array of tensors so we can use fancy indexing from numpy
        targets = [[tensor.cpu() for tensor in tensors] for tensors in targets] # Jona
        targets = np.array(targets, dtype='object')

        # Jona
        newclassifications = ()
        for tensor in classifications: newclassifications += (tensor.cpu(),)
        classifications = newclassifications

        # Compute the loss on every valid permutation of the targets
        # TODO think of a way to avoid this memory transfer but keep permutation indices synced with checkpoint
        losses = []
        for permutation in self.event_permutation_tensor.cpu().numpy():
            predictions = [tensor.cpu() for tensor in predictions] # Jona
            loss = tuple(jet_cross_entropy_loss(P, T, M, self.options.focal_gamma) +
                         self.particle_classification_loss(C, M)
                         for P, C, (T, M)
                         in zip(predictions, classifications, targets[permutation]))
            losses.append(torch.sum(torch.stack(loss), dim=0))

        losses = torch.stack(losses)

        # Squash the permutation losses into a single value.
        # Typically we just take the minimum, but it might
        # be interesting to examine other methods.
        combined_losses, index = losses.min(0)

        if self.options.combine_pair_loss.lower() == "mean":
            combined_losses = losses.mean(0)
            index = 0

        if self.options.combine_pair_loss.lower() == "softmin":
            weights = F.softmin(losses, 0)
            combined_losses = (weights * losses).sum(0)

        return combined_losses, index

    def training_step(self, batch: Tuple[Tuple[Tensor, Tensor], ...], batch_nb: int) -> Dict[str, Tensor]:
        # (source_data, source_mask), *targets = batch
        source_data, source_mask, target_data, target_mask = batch
        # process  
        source_data = source_data.float()
        targets = [(target_data[:,:self.num_children], target_mask[:,0]),(target_data[:,self.num_children:], target_mask[:,1])]

        # ===================================================================================================
        # Network Forward Pass
        # ---------------------------------------------------------------------------------------------------
        predictions = self.forward(source_data, source_mask)

        # Extract individual prediction data
        classifications = tuple(prediction[1] for prediction in predictions)
        predictions = tuple(prediction[0] for prediction in predictions)

        # ===================================================================================================
        # Initial log-likelihood loss for classification task
        # ---------------------------------------------------------------------------------------------------
        total_loss, best_indices = self.negative_log_likelihood(predictions, classifications, targets)

        # Log the classification loss to tensorboard.
        with torch.no_grad():
            self.log("loss/nll_loss", total_loss.mean())
            if torch.isnan(total_loss).any():
                raise ValueError("NLL Loss has diverged.")

        # Construct the newly permuted masks based on the minimal permutation found during NLL loss.
        permutations = self.event_permutation_tensor[best_indices].T
        masks = torch.stack([target[1] for target in targets])
        masks = torch.gather(masks, 0, permutations)

        # TODO Simple mean for speed
        total_loss = len(targets) * total_loss.sum() / masks.sum()
        # total_loss = total_loss.mean()

        self.log("loss/total_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx) -> Dict[str, np.float32]:
        # Run the base prediction step
        # (source_data, source_mask), *targets = batch
        source_data, source_mask, target_data, target_mask = batch
        # process  
        source_data = source_data.float()
        targets = [(target_data[:,:self.num_children], target_mask[:,0]),(target_data[:,self.num_children:], target_mask[:,1])]

        jet_predictions, particle_scores = self.predict_jets_and_particle_scores(source_data, source_mask)

        batch_size = source_data.shape[0]
        num_targets = len(targets)

        # Stack all of the targets into single array, we will also move to numpy for easier the numba computations.
        stacked_targets = np.zeros(num_targets, dtype=object)
        stacked_masks = np.zeros((num_targets, batch_size), dtype=np.bool)
        for i, (target, mask) in enumerate(targets):
            stacked_targets[i] = target.detach().cpu().numpy()
            stacked_masks[i] = mask.detach().cpu().numpy()

        metrics = self.evaluator.full_report_string(jet_predictions, stacked_targets, stacked_masks, prefix="Purity/")

        # Apply permutation groups for each target
        for target, prediction, decoder in zip(stacked_targets, jet_predictions, self.decoders):
            for indices in decoder.permutation_indices:
                if len(indices) > 1:
                    prediction[:, indices] = np.sort(prediction[:, indices])
                    target[:, indices] = np.sort(target[:, indices])

        metrics.update(self.compute_metrics(jet_predictions, particle_scores, stacked_targets, stacked_masks))

        for name, value in metrics.items():
            if not np.isnan(value):
                self.log(name, value)

        # Jona
        # Compute val_loss
        predictions     = self.forward(source_data,source_mask)
        classifications = tuple(prediction[1] for prediction in predictions)
        predictions     = tuple(prediction[0] for prediction in predictions)
        total_loss, best_indices = self.negative_log_likelihood(predictions,classifications,targets)
        permutations = self.event_permutation_tensor[best_indices].T
        masks        = torch.stack([target[1] for target in targets])
        masks        = torch.gather(masks, 0, permutations)
        total_loss   = len(targets) * total_loss.sum() / masks.sum()
        self.log("val_loss",total_loss)

        return metrics

    def configure_optimizers(self):

        optimizer = getattr(torch.optim, self.options.optimizer)
        if optimizer is None:
            print(f"Unable to load desired optimizer: {self.options.optimizer}.")
            print(f"Using pytorch AdamW as a default.")
            optimizer = torch.optim.AdamW

        decay_mask = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [param for name, param in self.named_parameters()
                           if not any(no_decay in name for no_decay in decay_mask)],
                "weight_decay": self.options.l2_penalty,
            },
            {
                "params": [param for name, param in self.named_parameters()
                           if any(no_decay in name for no_decay in decay_mask)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optimizer(optimizer_grouped_parameters, lr=self.options.learning_rate)

        if self.options.learning_rate_cycles < 1:
            scheduler = get_linear_schedule_with_warmup(
                 optimizer,
                 num_warmup_steps=self.options.warmup_steps,
                 num_training_steps=self.options.total_steps
             )
        else:
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.options.warmup_steps,
                num_training_steps=self.options.total_steps,
                num_cycles=self.options.learning_rate_cycles
            )

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def compute_metrics(self, jet_predictions, particle_scores, stacked_targets, stacked_masks):
        event_permutation_group = self.event_permutation_tensor.cpu().numpy()
        num_permutations = len(event_permutation_group)
        num_targets, batch_size = stacked_masks.shape
        particle_predictions = particle_scores >= 0.5

        # Compute all possible target permutations and take the best performing permutation
        # First compute raw_old accuracy so that we can get an accuracy score for each event
        # This will also act as the method for choosing the best permutation to compare for the other metrics.
        jet_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=np.bool)
        particle_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=np.bool)
        for i, permutation in enumerate(event_permutation_group):
            for j, (prediction, target) in enumerate(zip(jet_predictions, stacked_targets[permutation])):
                jet_accuracies[i, j] = np.all(prediction == target, axis=1)

            particle_accuracies[i] = stacked_masks[permutation] == particle_predictions

        jet_accuracies = jet_accuracies.sum(1)
        particle_accuracies = particle_accuracies.sum(1)

        # Select the primary permutation which we will use for all other metrics.
        chosen_permutations = self.event_permutation_tensor[jet_accuracies.argmax(0)].T
        chosen_permutations = chosen_permutations.cpu()
        permuted_masks = torch.gather(torch.from_numpy(stacked_masks), 0, chosen_permutations).numpy()

        # Compute final accuracy vectors for output
        num_particles = stacked_masks.sum(0)
        jet_accuracies = jet_accuracies.max(0)
        particle_accuracies = particle_accuracies.max(0)

        # Create the logging dictionaries
        metrics = {f"jet/accuracy_{i}_of_{j}": (jet_accuracies[num_particles == j] >= i).mean()
                   for j in range(1, num_targets + 1)
                   for i in range(1, j + 1)}

        metrics.update({f"particle/accuracy_{i}_of_{j}": (particle_accuracies[num_particles == j] >= i).mean()
                        for j in range(1, num_targets + 1)
                        for i in range(1, j + 1)})

        particle_scores = particle_scores.ravel()
        particle_targets = permuted_masks.ravel()
        particle_predictions = particle_predictions.ravel()

        self.particle_metrics = {
            "accuracy": sk_metrics.accuracy_score,
            "sensitivity": sk_metrics.recall_score,
            "specificity": lambda t, p: sk_metrics.recall_score(~t, ~p),
            "f_score": sk_metrics.f1_score
        }  

        self.particle_score_metrics = {
            "roc_auc": sk_metrics.roc_auc_score,
            "average_precision": sk_metrics.average_precision_score
        }
        
        for name, metric in self.particle_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_predictions)

        for name, metric in self.particle_score_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_scores)

        # Compute the sum accuracy of all complete events to act as our target for
        # early stopping, hyperparameter optimization, learning rate scheduling, etc.
        metrics["validation_accuracy"] = metrics[f"jet/accuracy_{num_targets}_of_{num_targets}"]

        return metrics
