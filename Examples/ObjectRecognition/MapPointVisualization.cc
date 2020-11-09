//
// Created by root on 2020/11/9.
//

#include <iostream>
#include <glog/logging.h>
#include <pangolin/gl/gldraw.h>
#include "GL/glu.h"
#include "GL/glut.h"
#include <Struct/PointCloudObject.h>
#include <ORBSLAM3/ViewerAR.h>
#include <Utility/FileIO.h>

using std::cout;
using std::endl;

typedef double M3DVector2d[2]; // 2D representations sometimes... (x,y) order
typedef double M3DVector3d[3]; // Vector of three doubles (x, y, z)
typedef double
    M3DVector4d[4]; // Yes, occasionaly we do need a trailing w component
typedef double
    M3DMatrix44d[16]; // A 4 x 4 matrix, column major (doubles) - OpenGL style

struct MapHandler3D_ShowMapPoint : public pangolin::Handler3D {
    MapHandler3D_ShowMapPoint(
        pangolin::OpenGlRenderState &cam_state,
        pangolin::AxisDirection enforce_up = pangolin::AxisNone,
        float trans_scale = 0.01f,
        float zoom_fraction = PANGO_DFLT_HANDLER3D_ZF)
        : Handler3D(cam_state, enforce_up, trans_scale, zoom_fraction) {
        m_view_state = &cam_state;
    };

    void Mouse(
        pangolin::View &display, pangolin::MouseButton button, int x, int y,
        bool pressed, int button_state) {
        pangolin::Handler3D::Mouse(
            display, button, x, y, pressed, button_state);
        if (pangolin::MouseButtonLeft == button) {
            if (button_state == 0) {
                m_is_left_button_down = false;
            } else if (button_state == 1) {
                m_left_button_pos.x() = x;
                m_left_button_pos.y() = y;
                m_is_left_button_down = true;
            } else if (
                button_state ==
                pangolin::MouseButtonLeft + pangolin::KeyModifierCtrl) {
                m_left_button_pos.x() = x;
                m_left_button_pos.y() = y;
                m_select_area_flag = true;
            } else if (button_state == pangolin::KeyModifierCtrl) {
                m_select_area_flag = false;
            }

            m_region_right_top.x() = x;
            m_region_right_top.y() = y;
            m_region_left_bottom.x() = x;
            m_region_left_bottom.y() = y;
            VLOG(5) << "mousex: " << m_region_left_bottom.x() << " "
                    << m_region_left_bottom.y() << m_region_right_top.x() << " "
                    << m_region_right_top.y();
        }

        if (pangolin::MouseButtonRight == button) {
            if (button_state == 0) {
                m_right_button_pos.x() = x;
                m_right_button_pos.y() = y;
                m_is_right_button_down = true;
            } else if (button_state == 1) {
                m_is_right_button_down = false;
            }
        }
    }

    void MouseMotion(pangolin::View &display, int x, int y, int button_state) {
        pangolin::Handler3D::MouseMotion(display, x, y, button_state);
        if (button_state ==
            pangolin::MouseButtonLeft + pangolin::KeyModifierCtrl) {
            m_region_right_top.x() = x;
            m_region_right_top.y() = y;
        }
    }

    int GetRightButtonPos(Eigen::Vector2d &right_button_pos) {
        right_button_pos = m_right_button_pos;
        return m_is_right_button_down;
    }

    int GetLeftButtonPos(Eigen::Vector2d &left_button_pos) {
        left_button_pos = m_left_button_pos;
        return m_is_left_button_down;
    }

    int GetSelected2DRegion(
        Eigen::Vector2d &region_left_bottom,
        Eigen::Vector2d &region_right_top) {
        region_left_bottom = m_region_left_bottom;
        region_right_top = m_region_right_top;
        return m_select_area_flag;
    }

    bool m_select_area_flag = false;
    bool m_is_left_button_down = false;
    bool m_is_right_button_down = false;
    pangolin::OpenGlRenderState *m_view_state;
    Eigen::Vector2d m_left_button_pos = Eigen::Vector2d::Zero();
    Eigen::Vector2d m_right_button_pos = Eigen::Vector2d::Zero();

    Eigen::Vector2d m_region_left_bottom = Eigen::Vector2d::Zero();
    Eigen::Vector2d m_region_right_top = Eigen::Vector2d::Zero();
};

class PointCloudModelViewer {
public:
    PointCloudModelViewer() = default;
    ;

    ~PointCloudModelViewer() = default;
    ;

    void Init();

    void DrawInit(const int w, const int h);

    void RegistPangolinUI();

    void Draw(const int w, const int h);

    void DrawPointCloud();

    void DrawModelKeyFrame();

    void DrawBoundingBox();

    void DrawMapPoints_SuperPoint(
        const std::set<ObjRecognition::MapPoint::Ptr> mappoint_picked);

    void DrawAxis();

    void Select2DRegion();

    void DrawSelected2DRegion(const int w, const int h);

    void Pick3DPointCloud();

    void DrawPointCloudPicked();

public:
    std::shared_ptr<ObjRecognition::Object> point_cloud_object_;

private:
    std::shared_ptr<pangolin::Var<bool>> show_boundingbox_;
    std::shared_ptr<pangolin::Var<bool>> show_pointcloud_;
    std::shared_ptr<pangolin::Var<bool>> m_show_keyframe;

    std::set<ObjRecognition::MapPoint::Ptr> m_pointClouds_picked;
    Eigen::Vector2d m_region_right_top = Eigen::Vector2d::Zero();
    Eigen::Vector2d m_region_left_bottom = Eigen::Vector2d::Zero();
    bool m_select_area_flag = false;

    pangolin::OpenGlRenderState s_cam_ShowMappoint;
    pangolin::View d_cam_ShowMappoint;
    std::unique_ptr<MapHandler3D_ShowMapPoint> mp_ShowMappoint_handler3d;

    std::unique_ptr<pangolin::Var<bool>> menu_stop;
};

void Project3DXYZToUV(
    M3DVector2d point_out, const M3DMatrix44d model_view_matrix,
    const M3DMatrix44d projection_matrix, const int view_port[4],
    const M3DVector3d point_in) {

    M3DVector3d point_result;
    gluProject(
        point_in[0], point_in[1], point_in[2], model_view_matrix,
        projection_matrix, view_port, &point_result[0], &point_result[1],
        &point_result[2]);

    point_out[0] = point_result[0];
    point_out[1] = point_result[1];
}

bool DropInArea(
    M3DVector3d x, const M3DMatrix44d model_view, const M3DMatrix44d proj,
    const int viewport[4], const Eigen::Vector2d &left_bottom,
    const Eigen::Vector2d &right_top) {
    M3DVector2d win_coord;

    Project3DXYZToUV(win_coord, model_view, proj, viewport, x);

    if ((win_coord[0] < left_bottom[0] && win_coord[0] < right_top[0]) ||
        (win_coord[0] > left_bottom[0] && win_coord[0] > right_top[0]))
        return false;

    if ((win_coord[1] < left_bottom[1] && win_coord[1] < right_top[1]) ||
        (win_coord[1] > left_bottom[1] && win_coord[1] > right_top[1]))
        return false;

    return true;
}

void PointCloudModelViewer::Pick3DPointCloud() {
    if (!m_select_area_flag)
        return;

    m_select_area_flag = false;

    GLint viewport[4];
    glPushMatrix();
    glGetIntegerv(GL_VIEWPORT, viewport);

    pangolin::OpenGlMatrix modelview_matrix =
        s_cam_ShowMappoint.GetModelViewMatrix();
    pangolin::OpenGlMatrix projection_matrix =
        s_cam_ShowMappoint.GetProjectionMatrix();

    auto mappoint_all = point_cloud_object_->GetPointClouds();
    for (auto const &it : mappoint_all) {
        Eigen::Vector3d pose = it->GetPose();
        GLdouble pose_arrary[3] = {pose[0], pose[1], pose[2]};
        if (DropInArea(
                pose_arrary, modelview_matrix.m, projection_matrix.m, viewport,
                m_region_left_bottom, m_region_right_top)) {

            if (!m_pointClouds_picked.count(it)) {
                auto observations = (*it).GetObservations();
                for (auto observation : observations) {
                    auto keyframe_id = observation.first;
                    auto idx = observation.second;
                    ObjRecognition::KeyFrame::Ptr keyframe;
                    for (auto keyframe1 : point_cloud_object_->GetKeyFrames()) {
                        if (keyframe1->GetID() == keyframe_id) {
                            keyframe = keyframe1;
                            break;
                        }
                    }

                    if (keyframe) {
                        cv::Mat img = keyframe->GetRawImage().clone();
                        std::vector<cv::KeyPoint> keypoints;
                        cv::KeyPoint keypoint = keyframe->GetKeyPoints()[idx];
                        keypoints.emplace_back(keypoint);
                        cv::drawKeypoints(img, keypoints, img);
                        cv::imwrite(
                            "/home/zhangye/data1/sfm/mappoints/" +
                                std::to_string((*it).GetIndex()) + "--" +
                                std::to_string(keyframe_id) + ".png",
                            img);
                    }
                }
                m_pointClouds_picked.insert(it);
                VLOG(0) << it->GetInfo();
            }
        }
    }
}

void PointCloudModelViewer::Init() {
    point_cloud_object_ = std::make_shared<ObjRecognition::Object>(0);
}

void PointCloudModelViewer::DrawMapPoints_SuperPoint(
    const std::set<ObjRecognition::MapPoint::Ptr> mappoint_picked) {
    glPointSize(4);
    glBegin(GL_POINTS);

    for (auto mappoint : point_cloud_object_->GetPointClouds()) {
        Eigen::Vector3d pos = mappoint->GetPose();
        if (mappoint_picked.count(mappoint)) {
            glColor3f(0.0, 0.0, 1.0);
            glVertex3f(pos.x(), pos.y(), pos.z());
        } else {
            glColor3f(0.0, 1.0, 0.0);
            glVertex3f(pos.x(), pos.y(), pos.z());
        }
    }
    glEnd();
}

void PointCloudModelViewer::Draw(const int w, const int h) {
    while (true) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam_ShowMappoint.Activate(s_cam_ShowMappoint);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        pangolin::glDrawAxis(0.6f);
        DrawSelected2DRegion(w, h);
        Pick3DPointCloud();
        DrawMapPoints_SuperPoint(m_pointClouds_picked);
        pangolin::FinishFrame();
        usleep(1000);
    }
}

void PointCloudModelViewer::DrawInit(const int w, const int h) {
    pangolin::CreatePanel("menu").SetBounds(
        0.0, 1.0, 0.0, pangolin::Attach::Pix(200));
    menu_stop =
        std::make_unique<pangolin::Var<bool>>("menu.Stop", false, false);

    std::function<void(void)> stop_selected_2d_region_callback =
        std::bind(&PointCloudModelViewer::Select2DRegion, this);
    pangolin::RegisterKeyPressCallback(
        pangolin::PANGO_CTRL + 's', stop_selected_2d_region_callback);

    s_cam_ShowMappoint = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.7, -3.5, 0, 0, 0, 0.0, -1.0, 0.0));
    mp_ShowMappoint_handler3d.reset(
        new MapHandler3D_ShowMapPoint(s_cam_ShowMappoint));
    d_cam_ShowMappoint =
        pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(200), 1.0, (float)w / h)
            .SetHandler(mp_ShowMappoint_handler3d.get());
    d_cam_ShowMappoint.show = true;
}

void PointCloudModelViewer::RegistPangolinUI() {
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.9, 1.0);
    show_boundingbox_ = std::make_shared<pangolin::Var<bool>>(
        "menu.Show BoundingBox", true, true);
    show_pointcloud_ = std::make_shared<pangolin::Var<bool>>(
        "menu.Show PointCloud", true, true);
    m_show_keyframe =
        std::make_shared<pangolin::Var<bool>>("menu.Show KeyFrame", true, true);

    std::function<void(void)> stop_select_2d_region_callback =
        std::bind(&PointCloudModelViewer::Select2DRegion, this);

    pangolin::RegisterKeyPressCallback(
        pangolin::PANGO_CTRL + 's', stop_select_2d_region_callback);
}

void PointCloudModelViewer::DrawPointCloud() {

    typedef std::shared_ptr<ObjRecognition::MapPoint> MPPtr;
    std::vector<MPPtr> &pointClouds = point_cloud_object_->GetPointClouds();
    glColor3f(0.0f, 1.0f, 0.0f);
    glPointSize(4.0);

    glBegin(GL_POINTS);
    for (int i = 0; i < pointClouds.size(); i++) {
        Eigen::Vector3f p = pointClouds[i]->GetPose().cast<float>();
        glVertex3f(p.x(), p.y(), p.z());
    }

    glEnd();
}

void DrawKeyFrame(
    const Eigen::Vector3f &p, const Eigen::Quaternionf &q,
    Eigen::Vector3f color = Eigen::Vector3f(0.0f, 1.0f, 1.0f),
    float cam_size = 0.1f) {
    // const Pose& pose = mp_map->GetCamera(m_current_frame_index).GetPose();
    Eigen::Vector3f center = p;

    const float length = cam_size;
    Eigen::Vector3f m_cam[5] = {Eigen::Vector3f(0.0f, 0.0f, 0.0f),
                                Eigen::Vector3f(-length, -length, length),
                                Eigen::Vector3f(-length, length, length),
                                Eigen::Vector3f(length, length, length),
                                Eigen::Vector3f(length, -length, length)};

    for (int i = 0; i < 5; ++i)
        m_cam[i] = q * m_cam[i] + center;
    // [0;0;0], [X;Y;Z], [X;-Y;Z], [-X;Y;Z], [-X;-Y;Z]
    glColor3fv(&color(0));
    glBegin(GL_LINE_LOOP);
    glVertex3fv(m_cam[0].data());
    glVertex3fv(m_cam[1].data());
    glVertex3fv(m_cam[4].data());
    glVertex3fv(m_cam[3].data());
    glVertex3fv(m_cam[2].data());
    glEnd();
    glBegin(GL_LINES);
    glVertex3fv(m_cam[0].data());
    glVertex3fv(m_cam[3].data());
    glEnd();
    glBegin(GL_LINES);
    glVertex3fv(m_cam[0].data());
    glVertex3fv(m_cam[4].data());
    glEnd();
    glBegin(GL_LINES);
    glVertex3fv(m_cam[1].data());
    glVertex3fv(m_cam[2].data());
    glEnd();
}

void PointCloudModelViewer::DrawModelKeyFrame() {

    auto allKFs = point_cloud_object_->GetKeyFrames();

    if (allKFs.empty())
        return;

    for (const auto &itKF : allKFs) {
        Eigen::Vector3d tcw;
        Eigen::Matrix3d Rcw;
        itKF->GetPose(Rcw, tcw);

        Eigen::Matrix3d Rwc = Rcw.transpose();
        Eigen::Vector3d twc = -Rwc * tcw;

        Eigen::Quaterniond Qwc(Rwc);
        DrawKeyFrame(twc.cast<float>(), Qwc.cast<float>());
    }
}

void PointCloudModelViewer::DrawAxis() {
    glEnable(GL_BLEND);
    pangolin::glDrawAxis(0.5f);
    glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
    pangolin::glDraw_z0(0.5f, 10);
    glDisable(GL_BLEND);
}

void PointCloudModelViewer::DrawBoundingBox() {

    std::vector<Eigen::Vector3d> boundingbox =
        point_cloud_object_->GetBoundingBox();

    glColor3f(1.0f, 1.0f, 0.0f);
    glPointSize(4.0);
    glBegin(GL_LINE_STRIP);

    glVertex3d(boundingbox[0].x(), boundingbox[0].y(), boundingbox[0].z());
    glVertex3d(boundingbox[1].x(), boundingbox[1].y(), boundingbox[1].z());

    glVertex3d(boundingbox[1].x(), boundingbox[1].y(), boundingbox[1].z());
    glVertex3d(boundingbox[2].x(), boundingbox[2].y(), boundingbox[2].z());

    glVertex3d(boundingbox[2].x(), boundingbox[2].y(), boundingbox[2].z());
    glVertex3d(boundingbox[3].x(), boundingbox[3].y(), boundingbox[3].z());

    glVertex3d(boundingbox[3].x(), boundingbox[3].y(), boundingbox[3].z());
    glVertex3d(boundingbox[0].x(), boundingbox[0].y(), boundingbox[0].z());

    glVertex3d(boundingbox[0].x(), boundingbox[0].y(), boundingbox[0].z());
    glVertex3d(boundingbox[4].x(), boundingbox[4].y(), boundingbox[4].z());

    glVertex3d(boundingbox[4].x(), boundingbox[4].y(), boundingbox[4].z());
    glVertex3d(boundingbox[7].x(), boundingbox[7].y(), boundingbox[7].z());
    glVertex3d(boundingbox[3].x(), boundingbox[3].y(), boundingbox[3].z());

    glVertex3d(boundingbox[7].x(), boundingbox[7].y(), boundingbox[7].z());
    glVertex3d(boundingbox[6].x(), boundingbox[6].y(), boundingbox[6].z());
    glVertex3d(boundingbox[2].x(), boundingbox[2].y(), boundingbox[2].z());

    glVertex3d(boundingbox[6].x(), boundingbox[6].y(), boundingbox[6].z());
    glVertex3d(boundingbox[5].x(), boundingbox[5].y(), boundingbox[5].z());
    glVertex3d(boundingbox[1].x(), boundingbox[1].y(), boundingbox[1].z());

    glVertex3d(boundingbox[5].x(), boundingbox[5].y(), boundingbox[5].z());
    glVertex3d(boundingbox[4].x(), boundingbox[4].y(), boundingbox[4].z());

    glEnd();
}

void PointCloudModelViewer::Select2DRegion() {
    Eigen::Vector2d region_left_bottom;
    Eigen::Vector2d region_right_top;
    m_select_area_flag = mp_ShowMappoint_handler3d->GetSelected2DRegion(
        m_region_left_bottom, m_region_right_top);
    VLOG(10) << "ctrl + s " << m_region_left_bottom.transpose() << " "
             << m_region_right_top.transpose();
}

void PointCloudModelViewer::DrawSelected2DRegion(const int w, const int h) {
    if (!m_select_area_flag)
        return;
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(200, w + 200, 0, h);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glEnable(GL_BLEND);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(1.0, 1.0, 0, 0.2);
    glRectf(
        m_region_left_bottom.x(), m_region_left_bottom.y(),
        m_region_right_top.x(), m_region_right_top.y());
    glEnable(GL_LINE_STIPPLE);

    glColor4f(1, 0, 0, 0.5);
    glLineStipple(3, 0xAAAA);
    glBegin(GL_LINE_STIPPLE);
    glVertex2f(m_region_left_bottom.x(), m_region_left_bottom.y());
    glVertex2f(m_region_right_top.x(), m_region_left_bottom.y());
    glVertex2f(m_region_right_top.x(), m_region_right_top.y());
    glVertex2f(m_region_left_bottom.x(), m_region_right_top.y());
    glEnd();
    glDisable(GL_LINE_STIPPLE);
    glDisable(GL_BLEND);
}

void PointCloudModelViewer::DrawPointCloudPicked() {

    glColor3f(1.0f, (51.0 / 255.0), (51.0 / 255.0));
    glPointSize(10.0);

    glBegin(GL_POINTS);
    for (const auto &MP : m_pointClouds_picked) {
        Eigen::Vector3f p = MP->GetPose().cast<float>();
        glVertex3f(p.x(), p.y(), p.z());
    }
    glEnd();
    //    m_pointClouds_picked.clear();
}

int main(int argc, char **argv) {

    if (argc < 2) {
        LOG(ERROR) << "./binPointCloudModel pointcloud.obj";
        return -1;
    }

    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InstallFailureSignalHandler();
    FLAGS_alsologtostderr = true;

    cv::FileStorage fsSettings(argv[1], cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    std::string model_file;
    fsSettings["mappoint_filename"] >> model_file;
    std::string data_path;
    fsSettings["saved_path"] >> data_path;
    std::string model_path = data_path + "/" + model_file;
    int w = 640;
    int h = 480;

    ObjRecognition::CameraIntrinsic::GetInstance().SetParameters(
        384.35386606462447, 384.9560729180638, 319.28590839603237,
        239.87334366520707, w, h);
    PointCloudModelViewer *viewer = new PointCloudModelViewer();
    viewer->Init();

    LoadPointCloudModel(model_path, viewer->point_cloud_object_);

    std::vector<ObjRecognition::MapPoint::Ptr> pointcloud =
        viewer->point_cloud_object_->GetPointClouds();
    LOG(INFO) << "point cloud size: " << pointcloud.size();
    pangolin::CreateWindowAndBind("MappointViewer", w + 200, h);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    viewer->DrawInit(w, h);
    //    viewer.RegistPangolinUI();
    viewer->Draw(w, h);
    return 0;
}
