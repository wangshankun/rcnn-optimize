#include "this_model_exe.h"
#include "idst_model_crypt.h"

int NetExecutor::Init(string configName)
{
    /* iaac */
    static IAACInfo ainfo = {
            .license_path = (char*)"/system/license.txt",
            .cid = 486310062,   // ingenic
            .fid = 1825793026,  // sentinel_dcr
            .sn = (char*)"7ce8a545522d860eca676d06d96a6e44",
            .send_and_recv = nullptr,
    };

    int ret = IAAC_Init(&ainfo);
    if (ret) {
            printf("%s:%d -> IAAC_Init error!\n", __func__, __LINE__);
            return -1;
    }

    //初始化网络
    Net* net = new Net();
    jzdl::BaseNet *base_net = jzdl::net_create();

    const char* model_path =  configName.c_str();
    if ( is_idst_encrypt_model(model_path) == 1 )
    {
        char key[64] = {};
        get_code_key(key);
        unsigned char* model_data = NULL;
        int size_out = 0;
        idst_rc4_de_file_to_mem(model_path, &model_data, &size_out, key);
        if(size_out > 0  && model_data != NULL)
        {
            if(base_net->load_model((const char*)model_data, true) == 0)
            {
                net->set_model_data(model_data);
            }
            else
            {
                printf("Model Load failed memory file %s failed! \n", model_path);
                return -1;
            }
        }
        else 
        {
            printf("Model Decrypted failed file %s failed! \n", model_path);
            return -1;
        }
    }
    else
    {
        if(base_net->load_model((const char*)model_path) == 0)
        {
        }
        else
        {
            printf("Model Load failed file %s failed! \n", model_path);
            return -1;
        }
    }

    net->base_net_  = base_net;
    net->threshold_ = 0.6;

    vector<uint32_t> input_shape = base_net->get_input_shape();
    net->model_in_w_ = input_shape[0];
    net->model_in_h_ = input_shape[1];
    net->model_in_c_ = input_shape[2];
    //生成priors,这个priors可以预生成保存到内存中不需要每次都计算
    generate_anchor(net->priors_, net->featuremap_size_, net->model_in_w_, net->model_in_h_);

    net->tracker_handle_= fusion_tracking::ped_tracker_create(0x111);
    fusion_tracking::ped_tracker_reset(net->tracker_handle_, "tracker");

    hdr_ = (void *)net;
    return 0;
}

int NetExecutor::Exec(vector<shared_ptr<HiTensor>> &input_vec,
                      vector<shared_ptr<OutputResult>> &output_vec)
{

    Net* net = (Net*)hdr_;
    jzdl::BaseNet* base_net = net->base_net_;
    int in_w = net->model_in_w_;
    int in_h = net->model_in_h_;
    int in_c = net->model_in_c_;

    jzdl::Mat<uint8_t> dst(in_w, in_h, in_c);
    for(int i = 0; i < input_vec.size(); i++)
    {
        int ori_w = input_vec[i]->w;
        int ori_h = input_vec[i]->h;

        if ( input_vec[i]->img_type == PIXEL_FORMAT_ABGR_8888 ||
             input_vec[i]->img_type == PIXEL_FORMAT_ARGB_8888 ) 
        {
            jzdl::Mat<uint8_t> src(ori_w, ori_h, in_c);
            jzdl::Mat<uint8_t> src_rgba(ori_w, ori_h, 4, input_vec[i]->data);
            bgra2bgr(src_rgba, src);
            prepare_data_resize_pad(src, dst, in_w, in_h);
        }
        else if(input_vec[i]->img_type == PIXEL_FORMAT_NV12)
        {
            uint8_t* rgb_p = (uint8_t*)malloc(ori_w * ori_h * in_c);
            NV12_TO_RGB24(input_vec[i]->data, rgb_p, ori_w, ori_h);
            jzdl::Mat<uint8_t> src(ori_w, ori_h, in_c, rgb_p);
            prepare_data_resize_pad(src, dst, in_w, in_h);
        }
        else if(input_vec[i]->img_type == PIXEL_FORMAT_RGB24_PACKAGE)
        {
            jzdl::Mat<uint8_t> src(ori_w, ori_h, in_c, input_vec[i]->data);
            prepare_data_resize_pad(src, dst, in_w, in_h);
        }
        jzdl::image_sub(dst, 128);
        //推理
        jzdl::Mat<int8_t> img(in_w, in_h, in_c, (int8_t*)dst.data);
        jzdl::Mat<float>  out;
        base_net->input(img);
        base_net->run(out);

        //格式化输出
        std::vector<float> scores;
        std::vector<float> boxes;
        int class_num = format_out(out, net->featuremap_size_, scores, boxes);
        vector<ObjMbox_t> detect_list;
        int num_anchors = net->priors_.size();
        generateBBox(detect_list, scores, net->priors_, boxes, net->threshold_, num_anchors, class_num, ori_w, ori_h, in_w, in_h);

        std::vector<std::shared_ptr<fusion_tracking::Detection>> tracker_list;

        for (int j = 0; j < detect_list.size(); j++) {
            auto box = detect_list[j];
	        fusion_tracking::Box2f normal_box(box.x0 / ori_w, 
                                                box.y0 / ori_h, 
                                                (box.x1 - box.x0) / ori_w, 
                                                (box.y1 - box.y0) / ori_h);
            //debug_print("detect_list box: %d %f %f %f %f %f\r\n",box.clsn, box.score, box.x0,box.y0,box.x1,box.y1);
            if(box.clsn == 1) 
            {
                std::shared_ptr<fusion_tracking::PersonDetection> this_detection = std::make_shared<fusion_tracking::PersonDetection>();
                this_detection->SetBoxNorm(normal_box, box.score);
                this_detection->set_category(fusion_tracking::ObjectCategory::CLASS_PEDESTRIAN);
                tracker_list.push_back(this_detection);
            }
            else if (box.clsn == 2)
            {
                std::shared_ptr<fusion_tracking::FaceDetection> this_detection = std::make_shared<fusion_tracking::FaceDetection>();
                this_detection->SetBoxNorm(normal_box, box.score);
                this_detection->set_category(fusion_tracking::ObjectCategory::CLASS_FACE);
                tracker_list.push_back(this_detection);
            }
            else if (box.clsn == 3)
            {
                //this_detection->set_category(fusion_tracking::ObjectCategory::CLASS_VEHICLE);
            }
            else if (box.clsn == 4)
            {
                std::shared_ptr<fusion_tracking::HeadDetection> this_detection = std::make_shared<fusion_tracking::HeadDetection>();
                this_detection->SetBoxNorm(normal_box, box.score);
                this_detection->set_category(fusion_tracking::ObjectCategory::CLASS_HEAD);
                tracker_list.push_back(this_detection);
            }
        }
        detect_list.clear();
        int64_t ts = input_vec[i]->time_stamp;
        std::vector<int64_t> out_keep_ts;  // no useful in our case
        fusion_tracking::ped_tracker_tracking_all_with_timestamp(net->tracker_handle_,
                                                        ts, tracker_list, out_keep_ts);
        std::vector<fusion_tracking::StructuralInfo> info_list;
        fusion_tracking::ped_tracker_get_person_info(net->tracker_handle_, ts, info_list);

        shared_ptr<OutputResult> out_res = shared_ptr<OutputResult>(new OutputResult);

        for (int i = 0; i < info_list.size(); ++i)
        {
            fusion_tracking::StructuralInfo track_info = info_list[i];
            DetBox box;
            box.score = track_info.detection_score;
            box.x0 = track_info.top_left_x * ori_w;
            box.x1 = track_info.bottom_right_x * ori_w;
            box.y0 = track_info.top_left_y * ori_h;
            box.y1 = track_info.bottom_right_y * ori_h;

            if (track_info.type == fusion_tracking::ClassType::TYPE_PERSON
                || track_info.type == fusion_tracking::ClassType::TYPE_HEAD)
            {
                box.clsn = 1;
                box.id = track_info.person_id;
            }
            else if(track_info.type == fusion_tracking::ClassType::TYPE_FACE)
            {
                box.clsn = 2;
                box.id = track_info.person_id;
            }
            else if(track_info.type == fusion_tracking::ClassType::TYPE_VEHICLE)
            {
                box.clsn = 3;
            }
            else
            {
                box.clsn = 0;
            }
            //debug_print("detect_list box: %d %f %f %f %f %f\r\n",box.clsn, box.score, box.x0,box.y0,box.x1,box.y1);
            out_res->boxes.push_back(box);
        }
        info_list.clear();

        output_vec.push_back(out_res);
    }

    return 0;
}

int NetExecutor::Destory()
{
    Net* net = (Net*)hdr_;
    fusion_tracking::ped_tracker_setEnd(net->tracker_handle_);
    fusion_tracking::ped_tracker_reset(net->tracker_handle_, "tracker");
    fusion_tracking::ped_tracker_destroy(&(net->tracker_handle_));
    delete net;
    return 0;
}

Executor* CreateNetExecutor()
{
    return new NetExecutor();
}