//
// Created by root on 2020/12/2.
//

#ifndef TEST2_MESH_H
#define TEST2_MESH_H
#include <glm/glm.hpp>
#include <GL/glew.h>
#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <vector>
using namespace std;

struct Vertex {
    // position
    glm::vec3 Position;
    // normal
    glm::vec3 Normal;
    // texCoords
    glm::vec2 TexCoords;
    // tangent
    glm::vec3 Tangent;
    // bitangent
    glm::vec3 Bitangent;
};

struct Texture {
    unsigned int id;
    string type;
    string path;
};

class Mesh {
public:
    // mesh Data
    vector<Vertex>       vertices;
    vector<unsigned int> indices;
    vector<Texture>      textures;
    unsigned int VAO;

    // constructor
    Mesh(vector<Vertex> vertices, vector<unsigned int> indices, vector<Texture> textures)
    {
        this->vertices = vertices;
        this->indices = indices;
        this->textures = textures;

        // now that we have all the required data, set the vertex buffers and its attribute pointers.
//        setupMesh();
    }

    void set_trans(Eigen::Matrix3d rot, Eigen::Vector3d offset) {
        printf("Warning, only cal Position, ignoring normal calculation and others\n");
        glm::mat<3, 3, double> mat;
        glm::vec<3, double> offset_glm;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                mat[j][i] = rot(i, j);
        offset_glm[0] = offset[0];
        offset_glm[1] = offset[1];
        offset_glm[2] = offset[2];

        for(auto &ver : this->vertices) {
            ver.Position = mat * ver.Position + offset_glm;
        }
    }

    // render the mesh
    void Draw()
    {
        // bind appropriate textures
//        unsigned int diffuseNr  = 1;
//        unsigned int specularNr = 1;
//        unsigned int normalNr   = 1;
//        unsigned int heightNr   = 1;
//        for(unsigned int i = 0; i < textures.size(); i++)
//        {
//            glActiveTexture(GL_TEXTURE0 + i); // active proper texture unit before binding
//            // retrieve texture number (the N in diffuse_textureN)
//            string number;
//            string name = textures[i].type;
//            if(name == "texture_diffuse")
//                number = std::to_string(diffuseNr++);
//            else if(name == "texture_specular")
//                number = std::to_string(specularNr++); // transfer unsigned int to stream
//            else if(name == "texture_normal")
//                number = std::to_string(normalNr++); // transfer unsigned int to stream
//            else if(name == "texture_height")
//                number = std::to_string(heightNr++); // transfer unsigned int to stream
//
//            // now set the sampler to the correct texture unit
//            glUniform1i(glGetUniformLocation(shader.ID, (name + number).c_str()), i);
//            // and finally bind the texture
//            glBindTexture(GL_TEXTURE_2D, textures[i].id);
//        }

        // draw mesh
//        glBindVertexArray(VAO);
//        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
//        glBindVertexArray(0);

        // always good practice to set everything back to defaults once configured.
//        glActiveTexture(GL_TEXTURE0);

        if (textures.size() > 0) {
            glColor3f(1.0f, 1.0f, 1.0f);
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
            glEnable(GL_TEXTURE_2D);
            glActiveTexture(GL_TEXTURE0 + 0); // active proper texture unit before binding
            printf("texture type %s\n", textures[0].type.c_str());
            printf("texture number %lu\n", textures.size());
            printf("texture id %u\n", textures[0].id);
            glBindTexture(GL_TEXTURE_2D, textures[0].id);

//            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        } else {
            glColor3f(1.0f, 0.0f, 0.0f);
        }

        glBegin(GL_TRIANGLES);
        Vertex ver;
//        printf("ind size %lu\n", indices.size());
        for (auto ind : indices) {
            ver = vertices[ind];
            glVertex3d(ver.Position[0], ver.Position[1], ver.Position[2]);
            if (textures.size() > 0) {
                printf("texCoords %f %f\n", ver.TexCoords[0], ver.TexCoords[1]);
                glTexCoord2f(ver.TexCoords[0], ver.TexCoords[1]);
            }
        }
        glEnd();
        if (textures.size() > 0) {
            glDisable(GL_TEXTURE_2D);
        }
//        printf("texture vs %lu\n", textures.size());
    }

private:
    // render data
    unsigned int VBO, EBO;

    // initializes all the buffer objects/arrays
//    void setupMesh()
//    {
//        // create buffers/arrays
//        glGenVertexArrays(1, &VAO);
//        glGenBuffers(1, &VBO);
//        glGenBuffers(1, &EBO);
//
//        glBindVertexArray(VAO);
//        // load data into vertex buffers
//        glBindBuffer(GL_ARRAY_BUFFER, VBO);
//        // A great thing about structs is that their memory layout is sequential for all its items.
//        // The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
//        // again translates to 3/2 floats which translates to a byte array.
//        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);
//
//        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
//        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
//
//        // set the vertex attribute pointers
//        // vertex Positions
//        glEnableVertexAttribArray(0);
//        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
//        // vertex normals
//        glEnableVertexAttribArray(1);
//        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
//        // vertex texture coords
//        glEnableVertexAttribArray(2);
//        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
//        // vertex tangent
//        glEnableVertexAttribArray(3);
//        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
//        // vertex bitangent
//        glEnableVertexAttribArray(4);
//        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Bitangent));
//
//        glBindVertexArray(0);
//    }
};
#endif //TEST2_MESH_H
