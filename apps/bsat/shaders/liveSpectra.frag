#version 410 core

uniform sampler2D tex;
uniform int       split; // The point to rotate the y axis at
uniform int       height; // The height of the texture in pixels

in vec2  TexCoord; // Normalized between 0 and 1
out vec4 FragColor;


void main()
{
    // Calculate the split point in texture coordinates
    /*
    float splitPoint = float(split)  / float(textureSize(tex, 0).y);
    
    if (TexCoord.y < splitPoint) {
    	FragColor = texture(tex, vec2(TexCoord.x, TexCoord.y + splitPoint));
    } else {
    	FragColor = texture(tex, vec2(TexCoord.x, TexCoord.y - splitPoint));
    }
    
    if (abs(TexCoord.y - splitPoint) < 0.005) {
        FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
*/
    FragColor = texture(tex, TexCoord);
}