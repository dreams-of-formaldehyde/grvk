#include "mantle/mantleExt.h"
#include "mantle_object.h"
#include "logger.h"

static inline VkSampleLocationEXT calculateSampleLocation(
    const GR_OFFSET2D* offset)
{
    return (VkSampleLocationEXT) {
        .x = (float)(offset->x + 8) / 16.f,
        .y = (float)(offset->y + 8) / 16.f,
    };
}

static 
GR_RESULT getGrResult(
    VkResult result)
{
    switch (result) {
    case VK_SUCCESS:
        return GR_SUCCESS;
    case VK_NOT_READY:
        return GR_NOT_READY;
    case VK_TIMEOUT:
        return GR_TIMEOUT;
    case VK_EVENT_SET:
        return GR_EVENT_SET;
    case VK_EVENT_RESET:
        return GR_EVENT_RESET;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
        return GR_ERROR_OUT_OF_MEMORY;
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return GR_ERROR_OUT_OF_GPU_MEMORY;
    case VK_ERROR_DEVICE_LOST:
        return GR_ERROR_DEVICE_LOST;
    case VK_ERROR_MEMORY_MAP_FAILED:
        return GR_ERROR_MEMORY_MAP_FAILED;
    default:
        break;
    }

    LOGW("unsupported result %d\n", result);
    return GR_ERROR_UNKNOWN;
}

static VkSampleCountFlags getVkSampleCountFlags(
    GR_UINT samples)
{
    switch (samples) {
    case 0:
    case 1:
        return VK_SAMPLE_COUNT_1_BIT;
    case 2:
        return VK_SAMPLE_COUNT_2_BIT;
    case 4:
        return VK_SAMPLE_COUNT_4_BIT;
    case 8:
        return VK_SAMPLE_COUNT_8_BIT;
    case 16:
        return VK_SAMPLE_COUNT_16_BIT;
    }

    LOGW("unsupported sample count %d\n", samples);
    return VK_SAMPLE_COUNT_1_BIT;
}

GR_RESULT GR_STDCALL grCreateAdvancedMsaaState(
    GR_DEVICE device,
    const GR_ADVANCED_MSAA_STATE_CREATE_INFO* pCreateInfo,
    GR_MSAA_STATE_OBJECT* pState)
{
    LOGT("%p %p %p\n", device, pCreateInfo, pState);
    GrDevice* grDevice = (GrDevice*)device;

    GrAdvancedMsaaStateObject* grMsaaStateObject = malloc(sizeof(GrAdvancedMsaaStateObject));
    if (pCreateInfo->pixelShaderSamples > 1) {
        LOGW("unhandled pixel shader sample count %d\n", pCreateInfo->pixelShaderSamples);
    }
    if (pCreateInfo->disableAlphaToCoverageDither) {
        LOGW("unhandled dither\n");
    }
    if (pCreateInfo->customSamplePatternEnable) {
        LOGW("custom sample pattern not supported\n");
    }
    // no need to handle depthStencil and color samples
    *grMsaaStateObject = (GrAdvancedMsaaStateObject) {
        .grObj = { GR_OBJ_TYPE_ADVANCED_MSAA_STATE_OBJECT, grDevice },
        .sampleCountFlags = getVkSampleCountFlags(pCreateInfo->coverageSamples),
        .sampleMask = pCreateInfo->sampleMask,
        .customSamplePatternEnabled = pCreateInfo->customSamplePatternEnable,
        .sampleLocations = { },
    };
    // grid size is always 2x2
    for (int i = 0; i < pCreateInfo->pixelShaderSamples; i++) {
        grMsaaStateObject->sampleLocations[i] = calculateSampleLocation(&pCreateInfo->customSamplePattern.topLeft[i]);
        grMsaaStateObject->sampleLocations[pCreateInfo->pixelShaderSamples + i] =
            calculateSampleLocation(&pCreateInfo->customSamplePattern.topRight[i]);
        grMsaaStateObject->sampleLocations[pCreateInfo->pixelShaderSamples * 2 + i] =
            calculateSampleLocation(&pCreateInfo->customSamplePattern.bottomLeft[i]);
        grMsaaStateObject->sampleLocations[pCreateInfo->pixelShaderSamples * 3 + i] =
            calculateSampleLocation(&pCreateInfo->customSamplePattern.bottomRight[i]);
    }

    *pState = (GR_MSAA_STATE_OBJECT)grMsaaStateObject;
    return GR_SUCCESS;
}

GR_RESULT GR_STDCALL grCreateFmaskImageView(
    GR_DEVICE device,
    const GR_FMASK_IMAGE_VIEW_CREATE_INFO* pCreateInfo,
    GR_IMAGE_VIEW* pView)
{
    LOGT("%p %p %p\n", device, pCreateInfo, pView);
    GrDevice* grDevice = (GrDevice*)device;
    VkImageView vkImageView = VK_NULL_HANDLE;

    // TODO validate parameters

    GrImage* grImage = (GrImage*)pCreateInfo->image;

    bool isArrayed = pCreateInfo->arraySize > 1;
    VkImageViewType imageViewType = isArrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;

    VkImageViewCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .image = grImage->image,
        .viewType = imageViewType,
        .format = grImage->format,
        .components = {
            .r = VK_COMPONENT_SWIZZLE_ZERO,
            .g = VK_COMPONENT_SWIZZLE_ZERO,
            .b = VK_COMPONENT_SWIZZLE_ZERO,
            .a = VK_COMPONENT_SWIZZLE_ZERO,
        },
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = pCreateInfo->baseArraySlice,
            .layerCount = pCreateInfo->arraySize,
        }
    };

    VkResult res = grDevice->vkd.vkCreateImageView(grDevice->device, &createInfo, NULL, &vkImageView);
    if (res != VK_SUCCESS) {
        LOGE("vkCreateImageView failed (%d)\n", res);
        return getGrResult(res);
    }

    GrImageView* grImageView = malloc(sizeof(GrImageView));
    *grImageView = (GrImageView) {
        .grObj = { GR_OBJ_TYPE_IMAGE_VIEW, grDevice },
        .imageView = vkImageView,
        .format = grImage->format,
        .usage = grImage->usage,
    };

    if (grDevice->descriptorBufferAllowPreparedImageView) {
        VkDescriptorImageInfo imageInfo = {
            .sampler = VK_NULL_HANDLE,
            .imageView = vkImageView,
            .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };

        VkDescriptorGetInfoEXT descriptorInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
            .pNext = NULL,
            .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .data = {
                .pSampledImage = &imageInfo,
            }
        };
        grDevice->vkd.vkGetDescriptorEXT(
            grDevice->device,
            &descriptorInfo,
            grDevice->descriptorBufferProps.sampledImageDescriptorSize,
            &grImageView->sampledDescriptor);
        if (grImageView->usage & VK_IMAGE_USAGE_STORAGE_BIT) {
            descriptorInfo.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorInfo.data.pStorageImage = &imageInfo;

            grDevice->vkd.vkGetDescriptorEXT(
                grDevice->device,
                &descriptorInfo,
                grDevice->descriptorBufferProps.storageImageDescriptorSize,
                &grImageView->storageDescriptor);
        }
    }
    *pView = (GR_IMAGE_VIEW)grImageView;
    return GR_SUCCESS;
}
