from model import Net
from design_sampling import DesignSampler, encode_data_list, TARGETS
from config import FLAGS
import torch
from saver import saver
from utils import load_pickle, get_gname

from os.path import join
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns

import learn2learn as l2l
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def visualize_model(dataset):
    dataset_by_gname = defaultdict(list)
    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        # data.data_id = i  # critical for exclude_adapted_data (rigorous testing of MAML)
        dataset_by_gname[get_gname(data)].append(data)

    if FLAGS.mode == 'vis_dse':
        _vis_dse(dataset, dataset_by_gname)
    elif FLAGS.mode == 'vis_trained_model':
        _vis_trained_model(dataset, dataset_by_gname)
    else:
        assert False


def _vis_dse(dataset, dataset_by_gname):
    col_path = f'{FLAGS.dse_collection_path}/obj/collection.pkl'
    ld = load_pickle(col_path)
    saver.log_info(f'Loaded {len(ld)} designs from dse process for {FLAGS.dse_kernel}')

    # bestdesign_path = f'{FLAGS.dse_collection_path}/obj/best_designs.pkl'
    # best_designs = load_pickle(bestdesign_path)
    # saver.log_info(f'Loaded {len(best_designs)} best designs from dse process for {FLAGS.dse_kernel}')

    labeled_designs = set()
    for data in tqdm(dataset):
        if data.gname == FLAGS.dse_kernel:
            s = _turn_point_into_str(_get_key_point_as_dict(data))
            # print(s)
            labeled_designs.add(s)

    saver.log_info(f'Found {len(labeled_designs)} labeled {FLAGS.dse_kernel} designs '
                   f'in dataset')
    assert len(labeled_designs) > 0, f'No {FLAGS.dse_kernel} designs in labeled dataset'

    embs = []
    matched = []
    matched_cnt = 0
    design_names = []
    # saver.log_info(f'Labeled designs provided from AutoDSS:')
    for point, emb in tqdm(ld):
        point_s = _turn_point_into_str(point)
        design_names.append(point_s)
        if point_s in labeled_designs:
            matched.append(True)
            matched_cnt += 1
            # saver.log_info(f'\t{point_s}')
        else:
            matched.append(False)
        embs.append(emb.detach().numpy())
    saver.log_info(f'{matched_cnt} designs out of {len(ld)} matched with labeled designs')

    X = np.vstack(embs)
    saver.log_info(f'X.shape {X.shape}')



    if FLAGS.vis_design_sampler:
        data_list = ld
        design_sampler = DesignSampler(data_list,
                                       sample_algo=FLAGS.adapt_designs_sample_algo,
                                       K=10,
                                       model=None, embs=X)
        for i in range(20):
            take, quality = design_sampler.sample_K()
            if FLAGS.save_sampled_designs:
                to_save = []
                for t in take:
                    to_save.append((t.gname, t.key))
                saver.save_dict(to_save, f'{FLAGS.dse_kernel}_{i}_sampled_{len(take)}_designs_keys')
        exit()



    feat_cols = ['feat' + str(i) for i in range(X.shape[1])]
    # print(feat_cols)
    df = pd.DataFrame(X, columns=feat_cols)
    df['matched'] = matched

    if FLAGS.clustering in ['K_Medoid', 'K_Means']:
        chosen_design_tf = [False] * X.shape[0]
        chosen_designs = []
        chosen_designs_dicts = []
        if FLAGS.clustering == 'K_Medoid':
            from sklearn_extra.cluster import KMedoids
            kmedoids = KMedoids(n_clusters=FLAGS.n_clusters, random_state=0,
                                metric=FLAGS.metric).fit(X)
            chosen_design_ids = kmedoids.medoid_indices_
                # print(ld[ind])
        elif FLAGS.clustering == 'K_Means':
            from sklearn.cluster import KMeans
            from sklearn.neighbors import NearestNeighbors
            kmeans = KMeans(init="k-means++", n_clusters=FLAGS.n_clusters,
                             random_state=0).fit(X)
            nbrs = NearestNeighbors(n_neighbors=1).fit(X)
            chosen_design_ids = []
            for ind in kmeans.cluster_centers_:
                _, chosen = nbrs.kneighbors(ind.reshape(1, len(ind)))
                assert chosen.shape == (1, 1)
                chosen_design_ids.append(chosen[0][0])
            pass
        else:
            assert False
        for ind in chosen_design_ids:
            chosen_design_tf[ind] = True
            chosen_designs.append(design_names[ind])
            chosen_designs_dicts.append(ld[ind])
        df[f'{FLAGS.clustering}_center'] = chosen_design_tf
        saver.log_info(f'Chosen designs according to {FLAGS.clustering} centers:')
        for dn in chosen_designs:
            saver.log_info(f'\t{dn}')
        saver.save_dict_as_pickle({'chosen_designs_dicts': chosen_designs_dicts,
                                   'chosen_designs_names': chosen_designs},
                                  f'{FLAGS.dse_kernel}_{FLAGS.clustering}_{FLAGS.n_clusters}_'
                                  f'from_{X.shape[0]}_designs')

    if X.shape[0] >= 10000:
        saver.log_info(f'Too many points -- skip TSNE:')
        return
    label_img = _gen_images_from_designs(df, design_sampler=None)
    saver.writer.add_embedding(X, metadata=design_names, label_img=label_img, tag='X')
    saver.log_info(f'Embedding saved for projector')
    tsne(df, feat_cols)


def _get_key_point_as_dict(data):
    li = data.key.decode('utf-8').split(':')[1].split('.')
    rtn = {}
    for s in li:
        ss = s.split('-')
        assert len(ss) == 2
        rtn[ss[0]] = ss[1]
    return rtn


def _turn_point_into_str(point):
    li = []
    for k, v in sorted(point.items()):
        li.append(f'{k}={v}')
    return ';'.join(li)


def _vis_trained_model(dataset, dataset_by_gname):
    setattr(FLAGS, 'target', ['perf', 'util-DSP', 'util-BRAM', 'util-LUT', 'util-FF'])
    model = Net(dataset.num_features).to(FLAGS.device)
    print(model)
    # print(model.state_dict().keys())
    assert FLAGS.load_model != 'None'
    if not FLAGS.only_pragma_nodes:
        saver.log_info(f'Loading model: {FLAGS.load_model}')
        ld = torch.load(FLAGS.load_model, map_location=FLAGS.device)

        if FLAGS.MAML and FLAGS.learning_algorithm == 'MAML':
            model = l2l.algorithms.MAML(model, lr=FLAGS.fast_lr, first_order=False)

        model.load_state_dict(ld)
        saver.log_info('Model loaded')
    else:
        saver.log_info('Skipping loading model since only pragma nodes')

    if FLAGS.vis_what == 'att':
        batch_size = 1
    elif FLAGS.vis_what == 'tsne':
        batch_size = 128
    else:
        raise NotImplementedError()

    only_kernel = getattr(FLAGS, 'only_kernel', 'None')
    if only_kernel != 'None':
        d_li = dataset_by_gname[only_kernel]
    else:
        d_li = dataset
    design_names, X, ys_dict = encode_data_list(
        d_li,
        getattr(FLAGS, 'only_kernel', 'None'),
        batch_size,
        model,
        FLAGS.vis_what,
        getattr(FLAGS, 'vis_emb', None),
        getattr(FLAGS, 'vis_emb_P_or_T', None),
        getattr(FLAGS, 'vis_y', None))

    # design_names2, X2, ys_dict2 = encode_data_list(
    #     dataset,
    #     '3mm',
    #     batch_size,
    #     model,
    #     FLAGS.vis_what,
    #     getattr(FLAGS, 'vis_emb', None),
    #     getattr(FLAGS, 'vis_emb_P_or_T', None),
    #     getattr(FLAGS, 'vis_y', None))

    # from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
    #
    # tot_cosine = cosine_similarity(np.vstack([X]))
    #
    # exit()


    saver.log_info(f'# of design_names: {len(design_names)}: {design_names[0:10]}')
    if FLAGS.vis_what == 'att':
        return

    saver.log_info(f'X.shape {X.shape}')
    assert X.shape[0] == len(design_names)

    feat_cols = ['feat' + str(i) for i in range(X.shape[1])]
    # print(feat_cols)
    df = pd.DataFrame(X, columns=feat_cols)
    assert len(design_names) == len(df)

    if FLAGS.vis_y == 'kernel_name':
        df['Kernel'] = ys_dict['Kernel']
    elif FLAGS.vis_y == 'target':
        for target_name in TARGETS:
            y = np.vstack(ys_dict[target_name])
            saver.log_info(f'{target_name} y.shape {y.shape}')

            if target_name == 'actual_perf':
                target_name = 'Latency'

            # if target_name == 'actual_perf':
            #     print(y)
            #     exit(-1)

            df[target_name] = y

            # print(y)
    else:
        raise NotImplementedError()

    # Visualize design sampler.
    if FLAGS.only_kernel != 'None' and FLAGS.vis_design_sampler:
        data_list = []
        for data in tqdm(dataset):
            # print('@@@ data.data_id', data.data_id, _get_gname(data))
            if FLAGS.only_kernel in data.gname:
                data.vis_id = len(data_list)
                data_list.append(data)
        assert len(data_list) == len(df), f'{len(data_list)} should be equal to {len(df)}'
        # exit()
        design_sampler = DesignSampler(data_list,
                                       sample_algo=FLAGS.adapt_designs_sample_algo,
                                       K=10,
                                       model=model)
    else:
        design_sampler = None

    # Images for embedding projector.
    label_img = _gen_images_from_designs(df, design_sampler)
    saver.writer.add_embedding(X, metadata=design_names, label_img=label_img, tag='X')

    print('Size of the dataframe: {}'.format(df.shape))

    embs = tsne(df, feat_cols)
    saver.save_dict_as_pickle({'tsne_embs': embs, 'df': df,
                               'feat_cols': feat_cols, 'design_names': design_names},
                              f'{FLAGS.vis_what}_{FLAGS.vis_y}')


def _gen_images_from_designs(df, design_sampler):
    images = []

    if design_sampler is not None:
        cmap = sns.color_palette("Set2", n_colors=2)
        take, _ = design_sampler.sample_K()
        take_ids = set([x.vis_id for x in take])
        for i in range(len(df)):
            if i in take_ids:
                cmap_id = 1  # 'red'
            else:
                cmap_id = 0  # 'green'
            images.append(torch.tensor(cmap[cmap_id]).view(3, 1, 1))
    else:
        _, cmap = _gen_tnames_cmap(df)
        if FLAGS.vis_y == 'kernel_name':
            kernels_seen = {}
            for kernel in df['Kernel']:
                kid = kernels_seen.get(kernel)
                if kid is None:
                    kid = len(kernels_seen)
                    kernels_seen[kernel] = kid
                images.append(torch.tensor(cmap[kid]).view(3, 1, 1))
            assert len(kernels_seen) == len(set(df['Kernel']))
        elif FLAGS.vis_y in ['kernel_name', 'matched']:
            k = FLAGS.vis_y
            if FLAGS.vis_y == 'kernel_name':
                k = 'Kernel'
            seen = {}
            for x in df[k]:
                xid = seen.get(x)
                if xid is None:
                    xid = len(seen)
                    seen[x] = xid
                images.append(torch.tensor(cmap[xid]).view(3, 1, 1))
            assert len(seen) == len(set(df[k]))
        elif FLAGS.vis_y == 'target':
            print(df['perf'].shape)
            for value in df['perf']:
                # print('$$$', value)
                # print('###', cmap(value))
                x = cmap(value)
                images.append(torch.tensor([x[0], x[1], x[2]]).view(3, 1, 1))
        else:
            raise NotImplementedError()

    rtn = torch.stack(images)
    saver.log_info(f'Images shape: {rtn.shape}')
    return rtn


perplexity = 25
n_iter = 5000


def tsne(df, feat_cols):
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
    X_to_fit = df[feat_cols].values
    # df.to_pickle('test_pickle.pkl')
    print(f'Size of X_to_fit: {X_to_fit.shape}')
    tsne_results = tsne.fit_transform(X_to_fit)
    df['x'] = tsne_results[:, 0]
    df['y'] = tsne_results[:, 1]

    sns.set(style="white", font_scale=2)

    # sns.set_style("whitegrid")
    # # sns.set(font_scale=2)

    tnames, cmap = _gen_tnames_cmap(df)

    for target_name in tnames:
        if target_name == 'actual_perf':
            target_name = 'Latency'
            # df[target_name] = 1e7/np.exp2(df[target_name]) # TODO: only true for certain case

        saver.log_info(f'min of {target_name}: {np.min(df[target_name])}')

        saver.log_info(f'max of {target_name}: {np.max(df[target_name])}')

        plt.figure(figsize=(16, 8))

        # try:

        # print('@@@', df[target_name])
        # print('###', cmap)

        if FLAGS.vis_y in ['kernel_name', 'matched']:

            style = FLAGS.vis_y
            if FLAGS.vis_y == 'kernel_name':
                style = 'Kernel'

            hue = target_name
            if FLAGS.mode == 'vis_dse' and FLAGS.clustering in ['K_Medoid', 'K_Means']:
                hue = f'{FLAGS.clustering}_center'

            sns_plot = sns.scatterplot(
                x="x", y="y",
                hue=hue,
                palette=cmap,
                data=df,
                s=75,
                # edgecolor=None,
                style=style
                # markers=['o', 's', 'X']
                # legend="brief",
                # alpha=0.3
            )

            # sns_plot.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
            sns_plot.legend(loc='best')
            sns_plot.set_title(target_name)

            cur_axes = plt.gca()
            cur_axes.axes.get_xaxis().set_visible(False)
            cur_axes.axes.get_yaxis().set_visible(False)
            plt.axis('off')

        elif FLAGS.vis_y == 'target':
            plt.scatter(df['x'], df['y'], label=None,
                        cmap=cmap,
                        # cmap='viridis',
                        c=df[target_name],
                        s=50, linewidth=0)
            # plt.axis(aspect='equal')
            # plt.xlabel('longitude')
            # plt.ylabel('latitude')
            plt.colorbar(label=target_name)
            # plt.clim(3, 7)

            # sns_plot = sns.scatterplot(
            #     x="x", y="y",
            #     hue=target_name,
            #     palette=cmap,
            #     data=df,
            #     s=100,
            #     edgecolor=None,
            #     # legend="brief",
            #     # alpha=0.3
            # )

            # sns_plot.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
            # sns_plot.legend(loc='center left')
            # sns_plot.set_title(target_name)

            # cur_axes = plt.gca()
            # cur_axes.axes.get_xaxis().set_visible(False)
            # cur_axes.axes.get_yaxis().set_visible(False)
            # plt.axis('off')

        else:
            raise NotImplementedError()

        kn = getattr(FLAGS, 'dse_kernel', None)
        s = ''
        if kn is not None:
            s = f' for the {kn} kernel'
        plt.title(f't-SNE visualization of design space{s}')

        # create_dir_if_not_exists(join(get_root_path(), DIR, KERNEL))
        save_path = join(saver.get_log_dir(),
                         f'tsne_{target_name}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        # save_path_npy = join(get_root_path(), DIR, KERNEL,
        #                      f'updated-dataset{PERF}-bottleneck-tsne-p{perplexity}-n{n_iter}.npy')
        # np.save(save_path_npy, tsne_results)
        # sns_plot.close()
        saver.log_info(f'Saved to {save_path}')
        # except Exception as e:
        #     saver.log_info(f'Error in {target_name}: {e}')

    return tsne_results


def _gen_tnames_cmap(df):
    if FLAGS.vis_y in ['kernel_name', 'matched']:
        s = FLAGS.vis_y
        if FLAGS.vis_y == 'kernel_name':
            s = 'Kernel'
            p = 'Paired'
        else:
            p = 'Set1'
        tnames = [s]
        cmap = sns.color_palette(p, n_colors=len(set(df[s])))
    elif FLAGS.vis_y == 'target':
        tnames = TARGETS
        # cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
        # cmap = sns.color_palette("rocket_r", as_cmap=True)
        # cmap = sns.color_palette("icefire", as_cmap=True)
        # cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)
        cmap = sns.color_palette("coolwarm", as_cmap=True)
    else:
        raise NotImplementedError()
    return tnames, cmap
