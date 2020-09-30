//
// Created by root on 2020/9/27.
//
#include <GL/gl.h>
#include <iostream>
#include <utility>
#include "ORBSLAM3/ScannerStruct/Struct.h"
void Triangle::SetVertex(
    Eigen::Vector3d v1, Eigen::Vector3d v2, Eigen::Vector3d v3) {
    m_v1 = std::move(v1);
    m_v3 = std::move(v3);
    m_v2 = std::move(v2);
}

int Scene::GetSceneHeight() const {
    return m_height;
}

bool Scene::GetIsChangingPlane() const {
    return m_is_changing_plane;
}

void Scene::SetIsChangingPlane(const bool &state) {
    m_is_changing_plane = state;
}

void Scene::SetSceneSize(int width, int height, int bar_width) {
    m_width = width;
    m_height = height;
    m_bar_width = bar_width;
}

int Scene::GetSceneBarWidth() const {
    return m_bar_width;
}

int Scene::GetSceneWidth() const {
    return m_width;
}

void Object::SetExist(bool exist) {
    m_exist = exist;
}

void Object::Reset() {
    minCornerPoint = Eigen::Vector3d::Zero();
    maxCornerPoint = Eigen::Vector3d::Zero();
    changePlaneOffset = 0.0;
    m_exist = false;
}

bool Object::IsExist() const {
    return m_exist;
}

void Object::MoveObject(float offset, int axies) {
    for (auto &i : m_vertex_list_p) {
        i[axies] += offset;
    }
    SetAllTriangles();
}

std::vector<Triangle> Object::GetAllTriangles() {
    return triangles;
}

void Object::SetVertexList(float vertex_list[8][3]) {
    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 3; j++) {
            m_vertex_list_p[i][j] = vertex_list[i][j];
        }
    }
    SetAllTriangles();
}

void Object::SetSize(const float side) {
    // draw on the plane
    float vertex_list[8][3] = {
        -side, -side - side, -side, side, -side - side, -side,
        -side, side - side,  -side, side, side - side,  -side,
        -side, -side - side, side,  side, -side - side, side,
        -side, side - side,  side,  side, side - side,  side,
    };

    // front, back, left, right, up, down
    GLint index_list[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3},
                               {4, 6}, {5, 7}, {0, 4}, {1, 5}, {7, 3}, {2, 6}};

    SetVertexList(vertex_list);
    SetIndexList(index_list);
}

void Object::SetIndexList(GLint index_list[12][2]) {
    for (size_t i = 0; i < 12; i++) {
        for (size_t j = 0; j < 2; j++) {
            m_index_list[i][j] = index_list[i][j];
        }
    }
}

void Object::SetChangeShapeOffset(float offset) {
    changePlaneOffset = offset;
}

float Object::GetChangeShapeOffset() {
    return changePlaneOffset;
}

void Object::ChangePlane(size_t planeNumber, float offset) {
    int axies;
    if (planeNumber == 0 || planeNumber == 1) {
        axies = 2;
        if (planeNumber == 0) {
            m_vertex_list_p[4][axies] += offset;
            m_vertex_list_p[5][axies] += offset;
            m_vertex_list_p[6][axies] += offset;
            m_vertex_list_p[7][axies] += offset;
        } else {
            m_vertex_list_p[0][axies] -= offset;
            m_vertex_list_p[1][axies] -= offset;
            m_vertex_list_p[2][axies] -= offset;
            m_vertex_list_p[3][axies] -= offset;
        }
    } else if (planeNumber == 2 || planeNumber == 3) {
        axies = 0;
        if (planeNumber == 2) {
            m_vertex_list_p[2][axies] -= offset;
            m_vertex_list_p[6][axies] -= offset;
            m_vertex_list_p[4][axies] -= offset;
            m_vertex_list_p[0][axies] -= offset;
        } else {
            m_vertex_list_p[3][axies] += offset;
            m_vertex_list_p[7][axies] += offset;
            m_vertex_list_p[5][axies] += offset;
            m_vertex_list_p[1][axies] += offset;
        }
    } else {
        axies = 1;
        if (planeNumber == 4) {
            m_vertex_list_p[2][axies] += offset;
            m_vertex_list_p[3][axies] += offset;
            m_vertex_list_p[6][axies] += offset;
            m_vertex_list_p[7][axies] += offset;
        } else {
            m_vertex_list_p[0][axies] -= offset;
            m_vertex_list_p[1][axies] -= offset;
            m_vertex_list_p[4][axies] -= offset;
            m_vertex_list_p[5][axies] -= offset;
        }
    }
    SetAllTriangles();
}

void Object::SetCornerPoint() {
    minCornerPoint = Eigen::Vector3d(
        m_vertex_list_p[0][0], m_vertex_list_p[0][1], m_vertex_list_p[0][2]);
    maxCornerPoint = Eigen::Vector3d(
        m_vertex_list_p[7][0], m_vertex_list_p[7][1], m_vertex_list_p[7][2]);
}

void Object::SetAllTriangles() {
    SetCornerPoint();
    Eigen::Vector3d point0 = Eigen::Vector3d(
        minCornerPoint(0), minCornerPoint(1), minCornerPoint(2));
    Eigen::Vector3d point1 = Eigen::Vector3d(
        maxCornerPoint(0), minCornerPoint(1), minCornerPoint(2));
    Eigen::Vector3d point2 = Eigen::Vector3d(
        minCornerPoint(0), maxCornerPoint(1), minCornerPoint(2));
    Eigen::Vector3d point3 = Eigen::Vector3d(
        maxCornerPoint(0), maxCornerPoint(1), minCornerPoint(2));
    Eigen::Vector3d point4 = Eigen::Vector3d(
        minCornerPoint(0), minCornerPoint(1), maxCornerPoint(2));
    Eigen::Vector3d point5 = Eigen::Vector3d(
        maxCornerPoint(0), minCornerPoint(1), maxCornerPoint(2));
    Eigen::Vector3d point6 = Eigen::Vector3d(
        minCornerPoint(0), maxCornerPoint(1), maxCornerPoint(2));
    Eigen::Vector3d point7 = Eigen::Vector3d(
        maxCornerPoint(0), maxCornerPoint(1), maxCornerPoint(2));
    // front
    t1.SetVertex(point6, point7, point4);
    t2.SetVertex(point7, point4, point5);
    triangle_plane[1] = {6, 7, 5, 4};
    triangle_plane[2] = {6, 7, 5, 4};
    // back
    t3.SetVertex(point2, point3, point0);
    t4.SetVertex(point3, point0, point1);
    triangle_plane[3] = {2, 3, 1, 0};
    triangle_plane[4] = {2, 3, 1, 0};
    // left
    t5.SetVertex(point2, point6, point4);
    t6.SetVertex(point2, point4, point0);
    triangle_plane[5] = {2, 0, 4, 6};
    triangle_plane[6] = {2, 0, 4, 6};
    // right
    t7.SetVertex(point3, point7, point5);
    t8.SetVertex(point3, point5, point1);
    triangle_plane[7] = {3, 7, 5, 1};
    triangle_plane[8] = {3, 7, 5, 1};
    // up
    t9.SetVertex(point3, point2, point6);
    t10.SetVertex(point3, point6, point0);
    triangle_plane[9] = {2, 3, 7, 6};
    triangle_plane[10] = {2, 3, 7, 6};
    // down
    t11.SetVertex(point0, point1, point4);
    t12.SetVertex(point1, point4, point5);
    triangle_plane[11] = {0, 1, 5, 4};
    triangle_plane[12] = {0, 1, 5, 4};

    triangles.clear();
    triangles = {t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12};
}

std::vector<Eigen::Vector3d> Triangle::GetVertex() {
    return {m_v1, m_v2, m_v3};
}

void Camera::SetCamPos(float x, float y, float z) {
    cameraPos_x = x;
    cameraPos_y = y;
    cameraPos_z = z;
}

void Camera::SetCamFront(float x, float y, float z) {
    cameraFront_x = x;
    cameraFront_y = y;
    cameraFront_z = z;
}

Eigen::Vector3d Camera::GetCamPos() const {
    return Eigen::Vector3d(cameraPos_x, cameraPos_y, cameraPos_z);
}

Eigen::Vector3d Camera::GetCamFront() const {
    return Eigen::Vector3d(cameraFront_x, cameraFront_y, cameraFront_z);
}

void MouseState::SetMousePoseX(int x) {
    mousepos_x = x;
}

float MouseState::GetMousePoseX() const {
    return mousepos_x;
}

void MouseState::SetMousePoseY(int y) {
    mousepos_y = y;
}

float MouseState::GetMousePoseY() const {
    return mousepos_y;
}

void MouseState::SetMouseLeftDown(bool state) {
    mouseLeftDown = state;
}

bool MouseState::IsMouseLeftDown() const {
    return mouseLeftDown;
}

void MouseState::SetMouseRightDown(bool state) {
    mouseRightDown = state;
}

bool MouseState::IsMouseRightDown() const {
    return mouseRightDown;
}

void MouseState::SetMouseWheelUp(bool state) {
    mouseWheelUp = state;
}

bool MouseState::IsMouseWheelUp() const {
    return mouseWheelUp;
}

void MouseState::SetMouseWheelDown(bool state) {
    mouseWheelDown = state;
}

bool MouseState::IsMouseWheelDown() const {
    return mouseWheelDown;
}

void MouseState::SetMouseAngleX(float x) {
    mouseangle_x = x;
}

float MouseState::GetMouseAngleX() const {
    return mouseangle_x;
}

void MouseState::SetMouseAngleY(float y) {
    mouseangle_y = y;
}

float MouseState::GetMouseAngleY() const {
    return mouseangle_y;
}

float MouseState::GetMouseWheelValue() const {
    return mouse_wheel_value;
}

void MouseState::SetMouseWheelValue(float value) {
    mouse_wheel_value = value;
}
