#include "mantle_internal.h"

// Extension Discovery Functions

GR_RESULT GR_STDCALL grGetExtensionSupport(
    GR_PHYSICAL_GPU gpu,
    const GR_CHAR* pExtName)
{
    LOGT("%p %s\n", gpu, pExtName);
    GrPhysicalGpu* grPhysicalGpu = (GrPhysicalGpu*)gpu;

    if (grPhysicalGpu == NULL) {
        return GR_ERROR_INVALID_HANDLE;
    }
    if (GET_OBJ_TYPE(grPhysicalGpu) != GR_OBJ_TYPE_PHYSICAL_GPU) {
        return GR_ERROR_INVALID_OBJECT_TYPE;
    }
    if (pExtName == NULL) {
        return GR_ERROR_INVALID_POINTER;
    }

    if (strcmp(pExtName, "GR_WSI_WINDOWS") == 0) {
        return GR_SUCCESS;
    } else if (strcmp(pExtName, "GR_BORDER_COLOR_PALETTE") == 0) {
        return GR_SUCCESS;
    } else if (strcmp(pExtName, "GR_DMA_QUEUE") == 0) {
        return GR_SUCCESS;
    } else if (strcmp(pExtName, "GR_ADVANCED_MSAA") == 0) {
        unsigned supportedExtensionCount = 0;
        if (vki.vkEnumerateDeviceExtensionProperties(grPhysicalGpu->physicalDevice, NULL, &supportedExtensionCount, NULL) != VK_SUCCESS) {
            LOGE("vkEnumerateDeviceExtensionProperties failed\n");
            return GR_UNSUPPORTED;
        }

        STACK_ARRAY(VkExtensionProperties, extensionProperties, 180, supportedExtensionCount);

        if (vki.vkEnumerateDeviceExtensionProperties(grPhysicalGpu->physicalDevice, NULL, &supportedExtensionCount, extensionProperties) != VK_SUCCESS) {
            STACK_ARRAY_FINISH(extensionProperties);
            LOGE("vkEnumerateDeviceExtensionProperties failed\n");
            return GR_UNSUPPORTED;
        }

        bool mixedMsaaSupported = false;
        for (unsigned i = 0; i < supportedExtensionCount; i++) {
            // check for VK_AMD_MIXED_ATTACHMENT_SAMPLES or VK_NV_FRAMEBUFFER_MIXED_SAMPLES extension
            // no need to have fragment sample mask AMD since it can be easily avoided
            if (strcmp(extensionProperties[i].extensionName, VK_NV_FRAMEBUFFER_MIXED_SAMPLES_EXTENSION_NAME) == 0 ||
                strcmp(extensionProperties[i].extensionName, VK_AMD_MIXED_ATTACHMENT_SAMPLES_EXTENSION_NAME) == 0) {
                mixedMsaaSupported = true;
                break;
            }
        }
        STACK_ARRAY_FINISH(extensionProperties);
        return mixedMsaaSupported ? GR_SUCCESS : GR_UNSUPPORTED;
    }

    LOGW("unsupported %s extension\n", pExtName);
    return GR_UNSUPPORTED;
}
