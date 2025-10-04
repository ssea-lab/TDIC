#coding=utf-8
# pylint: disable=import-error

import os
import sys
import time
import datetime

sys.path.append('/home/qibo/TDIC')

from absl import app
from absl import flags
from absl import logging

import torch
import torch.optim as optim
import numpy as np

from tqdm import tqdm

import model as MODELS
import data as DATA
import data_utils.loader as LOADER
import config.const as CONST
import metrics
import candidate_generator as cg

FLAGS = flags.FLAGS

# Minimal flags
flags.DEFINE_string('name', 'RunSimple', 'Experiment name.')
flags.DEFINE_enum('model', 'TDIC', ['MF',  'TDIC'], 'Model to train.')
flags.DEFINE_enum('dataset', 'ml10m', ['ml10m', 'nf'], 'Dataset.')
flags.DEFINE_integer('embedding_size', 64, 'Embedding size.')
flags.DEFINE_integer('epochs', 500, 'Training epochs.')
flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 5e-8, 'Weight decay.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('neg_sample_rate', 4, 'Negative sampling ratio.')
flags.DEFINE_bool('shuffle', True, 'Shuffle training set.')
flags.DEFINE_integer('num_workers', 4, 'DataLoader workers.')
flags.DEFINE_bool('use_gpu', True, 'Use GPU if available.')
flags.DEFINE_integer('gpu_id', 0, 'GPU id.')
flags.DEFINE_bool('cg_use_gpu', True, 'Use GPU for candidate generation.')
flags.DEFINE_integer('cg_gpu_id', 0, 'GPU ID for candidate generation.')
flags.DEFINE_string('output', '/home/amax/qibo/TDIC/output/', 'Output directory.')
flags.DEFINE_string('load_path', '', 'Dataset root directory. If empty, infer from --dataset.')
flags.DEFINE_enum('dis_loss', 'dcor', ['L1', 'L2', 'dcor'], 'discrepancy loss.')
flags.DEFINE_float('dis_pen', 0.01, 'discrepancy penalty.')
flags.DEFINE_float('int_weight', 0.1, 'interest weight.')
flags.DEFINE_float('pop_weight', 0.1, 'popularity weight.')
flags.DEFINE_float('tdic_weight', 0.1, 'TDIC weight.')
flags.DEFINE_integer('margin', 40, 'Margin for negative sampling.')
flags.DEFINE_integer('pool', 40, 'Pool for negative sampling.')
flags.DEFINE_float('margin_decay', 0.9, 'Decay of margin and pool.')
flags.DEFINE_float('loss_decay', 0.9, 'Decay of loss weights.')
flags.DEFINE_bool('adaptive', False, 'Adapt hyper-parameters during training.')
flags.DEFINE_multi_string('metrics', ['recall', 'hit_ratio', 'ndcg'], 'Metrics for evaluation.')
flags.DEFINE_multi_integer('topk', [20, 50], 'Topk for testing.')
flags.DEFINE_integer('num_test_users', 38016, 'Number of users for testing.')
flags.DEFINE_bool('test_after_train', True, 'Run test after training.')
flags.DEFINE_string('workspace', '', 'Workspace directory (auto-set from output).')
flags.DEFINE_bool('use_early_stop', True, 'Use early stopping during training.')
flags.DEFINE_integer('val_after_epochs', 5, 'Validate every N epochs.')
flags.DEFINE_integer('patience', 3, 'Patience for learning rate reduction.')
flags.DEFINE_integer('es_patience', 5, 'Patience for early stopping.')
flags.DEFINE_float('min_lr', 1e-5, 'Minimum learning rate.')
flags.DEFINE_string('watch_metric', 'recall', 'Metric to watch for early stopping (recall, hit_ratio, ndcg).')


def _set_load_path(flags_obj):
    if getattr(flags_obj, 'load_path', ''):
        return flags_obj.load_path
    if flags_obj.dataset == 'ml10m':
        return CONST.ml10m
    elif flags_obj.dataset == 'nf':
        return CONST.nf
    return CONST.ml10m


def _prepare_workspace(flags_obj):
    date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists(flags_obj.output):
        os.makedirs(flags_obj.output, exist_ok=True)
    workspace = os.path.join(flags_obj.output, f"{flags_obj.name}_{date_time}")
    os.makedirs(workspace, exist_ok=True)
    log_dir = os.path.join(workspace, 'log')
    os.makedirs(log_dir, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file(flags_obj.name + '.log', log_dir)
    return workspace


def _get_device(flags_obj):
    if flags_obj.use_gpu and torch.cuda.is_available():
        return torch.device(f'cuda:{flags_obj.gpu_id}')
    return torch.device('cpu')


def _load_dataset_info(flags_obj):
    print("Loading dataset info...")
    loader = LOADER.CooLoader(flags_obj)
    coo = loader.load(CONST.train_coo_record)
    n_user, n_item = coo.shape
    print(f"Dataset loaded: {n_user} users, {n_item} items")
    return n_user, n_item


def _build_dataloader(flags_obj):
    print("Building dataloader...")

    if flags_obj.model == 'MF':
        loader = DATA.FactorizationDataProcessor.get_blend_pair_dataloader
    else:
        loader = DATA.FactorizationDataProcessor.get_TDIC_dataloader

    # We need a minimal DatasetManager replacement providing coo/skew/popularity to factories.
    class _DM:  # very small holder
        def get_popularity(self):
            return self.popularity
        pass
    dm = _DM()

    # Fill required fields
    coo_loader = LOADER.CooLoader(flags_obj)
    dm.coo_record = coo_loader.load(CONST.train_coo_record)
    dm.skew_coo_record = coo_loader.load(CONST.train_skew_coo_record)
    if flags_obj.model in ['TDIC']:
        npy_loader = LOADER.NpyLoader(flags_obj)
        dm.popularity = npy_loader.load(CONST.popularity)
    print("Creating dataloader...")
    dataloader = loader(flags_obj, dm)
    print(f"Dataloader created with {len(dataloader)} batches")
    return dataloader


def _build_model(flags_obj, n_user, n_item, device):
    print(f"Building {flags_obj.model} model...")
    if flags_obj.model == 'MF':
        net = MODELS.MF(n_user, n_item, flags_obj.embedding_size)
    elif flags_obj.model == 'TDIC':
        net = MODELS.TDIC(
            n_user,
            n_item,
            flags_obj.embedding_size,
            flags_obj.dis_loss,
            flags_obj.dis_pen,
            flags_obj.int_weight,
            flags_obj.pop_weight,
            flags_obj.tdic_weight,
        )
    net = net.to(device)
    print(f"Model created and moved to {device}")
    return net


def _bpr_loss(p_score, n_score):
    return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))


def _train_epoch(flags_obj, epoch, model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batch = len(dataloader)
    start = time.time()

    print(f"Starting epoch {epoch} with {num_batch} batches...")
    for batch_idx, sample in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        optimizer.zero_grad()
        if flags_obj.model == 'MF':
            user, item_p, item_n = sample
            user = user.to(device)
            item_p = item_p.to(device)
            item_n = item_n.to(device)
            p_score, n_score = model.pair_forward(user, item_p, item_n)
            loss = _bpr_loss(p_score, n_score)
        else:  # TDIC
            if len(sample) == 5:  # 包含时间戳的新格式
                user, item_p, item_n, mask, timestamp = sample
                user = user.to(device)
                item_p = item_p.to(device)
                item_n = item_n.to(device)
                mask = mask.to(device)
                timestamp = timestamp.to(device)
                loss = model(user, item_p, item_n, mask, timestamp)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        if batch_idx % 100 == 0:  # 每100个batch打印一次
            logging.info('epoch %d: running loss = %.6f', epoch, total_loss / (batch_idx + 1))

    cost = time.time() - start
    logging.info('epoch %d: total loss = %.6f, time = %.2fs', epoch, total_loss, cost)


def main(argv):
    print("Starting training...")
    flags_obj = FLAGS

    # Route dataset path into flags for loader helpers
    flags_obj.load_path = _set_load_path(flags_obj)
    print(f"Using data path: {flags_obj.load_path}")

    workspace = _prepare_workspace(flags_obj)
    device = _get_device(flags_obj)
    print(f"Using device: {device}")

    n_user, n_item = _load_dataset_info(flags_obj)
    dataloader = _build_dataloader(flags_obj)

    net = _build_model(flags_obj, n_user, n_item, device)
    optimizer = optim.Adam(net.parameters(), lr=flags_obj.lr, weight_decay=flags_obj.weight_decay, betas=(0.5, 0.99), amsgrad=True)
    
    # Early stop setup
    if flags_obj.use_early_stop:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=flags_obj.patience, 
            min_lr=flags_obj.min_lr, factor=0.5, verbose=True
        )
        best_metric = -1.0
        best_epoch = -1
        no_improve_count = 0
        print("Early stopping enabled with validation every {} epochs".format(flags_obj.val_after_epochs))
    
    print("Starting training loop...")

    for epoch in range(flags_obj.epochs):
        _train_epoch(flags_obj, epoch, net, dataloader, optimizer, device)
        
        # Validation and early stopping
        if flags_obj.use_early_stop and (epoch + 1) % flags_obj.val_after_epochs == 0:
            print(f"\n--- Validation at epoch {epoch + 1} ---")
            val_results = _validate_model(flags_obj, net, n_user, n_item, device)
            watch_value = val_results[flags_obj.watch_metric]
            
            # Update learning rate
            scheduler.step(watch_value)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")
            
            # Check for improvement
            if watch_value > best_metric:
                best_metric = watch_value
                best_epoch = epoch
                no_improve_count = 0
                print(f"New best {flags_obj.watch_metric}: {best_metric:.4f} at epoch {epoch + 1}")
                
                # Save best model
                ckpt_dir = os.path.join(workspace, 'ckpt')
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(net.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
                print("Best model saved!")
            else:
                no_improve_count += 1
                print(f"No improvement for {no_improve_count} validations")
                
                # Early stopping
                if no_improve_count >= flags_obj.es_patience:
                    print(f"Early stopping triggered after {no_improve_count} validations without improvement")
                    break

    # Save final checkpoint
    ckpt_dir = os.path.join(workspace, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(ckpt_dir, f'epoch_{epoch}.pth'))
    
    if flags_obj.use_early_stop and best_epoch >= 0:
        print(f"\nTraining completed! Best {flags_obj.watch_metric}: {best_metric:.4f} at epoch {best_epoch + 1}")
        print("Loading best model for testing...")
        net.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best_model.pth')))
    else:
        print("Training completed!")
    
    # Test the model
    if flags_obj.test_after_train:
        print("Starting testing...")
        # Test for each topk value separately
        for topk in flags_obj.topk:
            print(f"\n--- Testing with Top-{topk} ---")
            test_results = _test_model(flags_obj, net, n_user, n_item, device, topk)
            print(f"Top-{topk} Results:")
            for metric, value in test_results.items():
                print(f"  {metric}: {value:.4f}")


def _test_model(flags_obj, model, n_user, n_item, device, topk):
    """Test the trained model and return results for a specific topk."""
    print("Preparing test data...")
    
    # Create minimal dataset manager for testing
    class _TestDM:
        pass
    dm = _TestDM()
    dm.n_user = n_user
    
    # Load test data
    coo_loader = LOADER.CooLoader(flags_obj)
    dm.coo_record = coo_loader.load(CONST.train_coo_record)
    
    # Create test dataloader
    test_dataloader, topk_margin = DATA.CGDataProcessor.get_dataloader(flags_obj, 'test')
    print(f"Test dataloader has {len(test_dataloader)} batches, topk_margin: {topk_margin}")
    print(f"Total test users available: {len(test_dataloader) * flags_obj.batch_size}")
    
    # Create judger - set workspace from output
    flags_obj.workspace = flags_obj.output
    judger = metrics.Judger(flags_obj, dm, topk)
    judger.metrics = flags_obj.metrics
    
    # Generate embeddings for candidate generation
    print("Generating embeddings...")
    model.eval()
    with torch.no_grad():
        if flags_obj.model == 'MF':
            item_embeddings = model.get_item_embeddings()
            user_embeddings = model.get_user_embeddings()
        else:  #TDIC
            item_embeddings = model.get_item_embeddings()
            user_embeddings = model.get_user_embeddings()
    
    # Create candidate generator
    generator = cg.FaissInnerProductMaximumSearchGenerator(flags_obj, item_embeddings)
    
    # Test
    print(f"Running evaluation for Top-{topk}...")
    results = {metric: 0.0 for metric in flags_obj.metrics}
    real_num_test_users = 0
    cg_topk = topk + topk_margin
    
    # Calculate how many batches we need to process
    num_test_batches = (flags_obj.num_test_users + flags_obj.batch_size - 1) // flags_obj.batch_size
    processed_users = 0
    total_available_users = len(test_dataloader) * flags_obj.batch_size
    actual_test_users = min(flags_obj.num_test_users, total_available_users)
    print(f"Will test {actual_test_users} users out of {total_available_users} available")
    
    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_dataloader, desc="Testing")):
            if processed_users >= actual_test_users:
                print(f"\nStopping at batch {batch_count}, processed {processed_users} users")
                break
                
            try:
                users, train_pos, test_pos, num_test_pos = data
                users = users.squeeze()
                
                # Generate candidates
                user_emb = user_embeddings[users]
                items = generator.generate(user_emb, cg_topk)
                
                # Filter training history
                items = _filter_history(items, train_pos, topk)
                
                # Evaluate
                batch_results, valid_num_users = judger.judge(items, test_pos, num_test_pos)
                real_num_test_users += valid_num_users
                processed_users += len(users)  # Count processed users
                
                # Update results
                for metric, value in batch_results.items():
                    results[metric] += value
                    
            except Exception as e:
                print(f"\nError at batch {batch_count}: {e}")
                print(f"Processed {processed_users} users before error")
                break
    
    # Average results
    if real_num_test_users > 0:
        for metric in results:
            if metric in ['recall', 'hit_ratio', 'ndcg']:
                results[metric] /= real_num_test_users
    
    return results


def _validate_model(flags_obj, model, n_user, n_item, device):
    """Validate the model on validation set."""
    # Create minimal dataset manager for validation
    class _ValDM:
        pass
    dm = _ValDM()
    dm.n_user = n_user
    
    # Load validation data
    coo_loader = LOADER.CooLoader(flags_obj)
    dm.coo_record = coo_loader.load(CONST.train_coo_record)
    
    # Create validation dataloader
    val_dataloader, topk_margin = DATA.CGDataProcessor.get_dataloader(flags_obj, 'val')
    
    # Create judger
    flags_obj.workspace = flags_obj.output
    judger = metrics.Judger(flags_obj, dm, max(flags_obj.topk))
    judger.metrics = flags_obj.metrics
    
    # Generate embeddings
    model.eval()
    with torch.no_grad():
        if flags_obj.model == 'MF':
            item_embeddings = model.get_item_embeddings()
            user_embeddings = model.get_user_embeddings()
        else:  #TDIC
            item_embeddings = model.get_item_embeddings()
            user_embeddings = model.get_user_embeddings()
    
    # Create candidate generator
    generator = cg.FaissInnerProductMaximumSearchGenerator(flags_obj, item_embeddings)
    
    # Validate
    results = {metric: 0.0 for metric in flags_obj.metrics}
    real_num_val_users = 0
    max_topk = max(flags_obj.topk)
    cg_topk = max_topk + topk_margin
    
    # Use a smaller number of users for validation (faster)
    num_val_users = min(5000, len(val_dataloader) * flags_obj.batch_size)
    num_val_batches = (num_val_users + flags_obj.batch_size - 1) // flags_obj.batch_size
    processed_users = 0
    
    with torch.no_grad():
        for batch_count, data in enumerate(val_dataloader):
            if processed_users >= num_val_users or batch_count >= num_val_batches:
                break
                
            users, train_pos, test_pos, num_test_pos = data
            users = users.squeeze()
            
            # Generate candidates
            user_emb = user_embeddings[users]
            items = generator.generate(user_emb, cg_topk)
            
            # Filter training history
            items = _filter_history(items, train_pos, max_topk)
            
            # Evaluate
            batch_results, valid_num_users = judger.judge(items, test_pos, num_test_pos)
            real_num_val_users += valid_num_users
            processed_users += len(users)
            
            # Update results
            for metric, value in batch_results.items():
                results[metric] += value
    
    # Average results
    if real_num_val_users > 0:
        for metric in results:
            if metric in ['recall', 'hit_ratio', 'ndcg']:
                results[metric] /= real_num_val_users
    
    # Print validation results
    print("Validation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    return results


def _filter_history(items, train_pos, max_topk):
    """Filter out items that user has already interacted with."""
    filtered_items = []
    for i in range(len(items)):
        # Remove items that are in training history
        valid_items = items[i][~np.isin(items[i], train_pos[i])]
        # Take top-k
        filtered_items.append(valid_items[:max_topk])
    return np.stack(filtered_items, axis=0)


if __name__ == '__main__':
    app.run(main)
