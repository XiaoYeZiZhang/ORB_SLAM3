#ifndef ORB_SLAM3_STRUCT_H
#define ORB_SLAM3_STRUCT_H
#include <Eigen/Dense>
#include <vector>
#include <map>
class Triangle {
public:
    Triangle() {
        m_v1 = Eigen::Vector3d::Zero();
        m_v3 = Eigen::Vector3d::Zero();
        m_v2 = Eigen::Vector3d::Zero();
    }
    void SetVertex(Eigen::Vector3d v1, Eigen::Vector3d v2, Eigen::Vector3d v3);
    std::vector<Eigen::Vector3d> GetVertex();

private:
    Eigen::Vector3d m_v1;
    Eigen::Vector3d m_v2;
    Eigen::Vector3d m_v3;
};

class Scene {
public:
    Scene() {
        m_width = 100;
        m_height = 100;
        m_is_changing_plane = false;
        m_bar_width = 0;
    }

    void SetSceneSize(int width, int height, int bar_width);
    int GetSceneWidth() const;
    int GetSceneHeight() const;
    int GetSceneBarWidth() const;
    bool GetIsChangingPlane() const;
    void SetIsChangingPlane(const bool &state);

private:
    int m_width;
    int m_height;
    int m_bar_width;
    bool m_is_changing_plane;
};

class Object {
public:
    Object() {
        minCornerPoint = Eigen::Vector3d::Zero();
        maxCornerPoint = Eigen::Vector3d::Zero();
        changePlaneOffset = 0.0;
        m_exist = false;
    }
    void SetExist(bool exist);
    bool IsExist() const;
    void SetCornerPoint();
    void Reset();
    void SetSize(float side);
    void MoveObject(float offset, int axies);
    void SetVertexList(float vertex_list[8][3]);
    void SetIndexList(GLint index_list[12][2]);
    void SetAllTriangles();
    void SetChangeShapeOffset(float offset_x);
    float GetChangeShapeOffset();
    void ChangePlane(size_t planeNumber, float offset);
    std::vector<Triangle> GetAllTriangles();

    float m_vertex_list_p[8][3];
    GLint m_index_list[12][2];
    std::map<int, std::pair<std::vector<int>, Eigen::Vector3d>> triangle_plane;
#if 0
    std::map<int, std::map<int, std::vector<Eigen::Vector3d>>>
        small_triangle_plane;
#endif
    int minTriangleIndex = -1;
    Eigen::Vector3d minCornerPoint;
    Eigen::Vector3d maxCornerPoint;

private:
    float changePlaneOffset;
    bool m_exist;
    Triangle t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12;
    std::vector<Triangle> triangles;
};

class Camera {
public:
    Camera() {
        cameraPos_x = 0.0;
        cameraPos_y = 0.0;
        cameraPos_z = 0.0;
        cameraFront_x = 0.0;
        cameraFront_y = 0.0;
        cameraFront_z = 0.0;
        cameraUp_x = 0.0;
        cameraUp_y = 0.0;
        cameraUp_z = 0.0;
    }

    Eigen::Vector3d GetCamPos() const;
    void SetCamPos(float x, float y, float z);
    Eigen::Vector3d GetCamFront() const;
    void SetCamFront(float x, float y, float z);

private:
    float cameraPos_x;
    float cameraPos_y;
    float cameraPos_z;
    float cameraFront_x;
    float cameraFront_y;
    float cameraFront_z;
    float cameraUp_x;
    float cameraUp_y;
    float cameraUp_z;
};

class MouseState {
public:
    MouseState() {
        mouseLeftDown = false;
        mouseRightDown = false;
        mouseWheelUp = false;
        mouseWheelDown = false;
        mousepos_x = 0.0;
        mousepos_y = 0.0;
        mouse_wheel_value = 1.0;
        mouseangle_x = 0.0;
        mouseangle_y = 0.0;
    }

    void SetMouseAngleX(float x);
    float GetMouseAngleX() const;
    void SetMouseAngleY(float y);
    float GetMouseAngleY() const;
    void SetMousePoseX(int x);
    float GetMousePoseX() const;
    void SetMousePoseY(int y);
    float GetMousePoseY() const;
    void SetMouseLeftDown(bool state);
    bool IsMouseLeftDown() const;
    void SetMouseRightDown(bool state);
    bool IsMouseRightDown() const;
    void SetMouseWheelUp(bool state);
    bool IsMouseWheelUp() const;
    void SetMouseWheelDown(bool state);
    bool IsMouseWheelDown() const;
    float GetMouseWheelValue() const;
    void SetMouseWheelValue(float value);

private:
    bool mouseLeftDown;
    bool mouseRightDown;
    bool mouseWheelUp;
    bool mouseWheelDown;
    float mousepos_x;
    float mousepos_y;
    float mouseangle_x;
    float mouseangle_y;
    float mouse_wheel_value;
};
#endif // ORB_SLAM3_STRUCT_H
