local classic = require 'classic'
local image = require 'image'

-- Detect QT for image display
local qt = pcall(require, 'qt')

local Display = classic.class('Display')


function Display:_init(opt, display)
  self.opt = opt
  -- Activate display if using QT
  self.zoom = opt.zoom
  self.window = qt and image.display({image=display, zoom=self.zoom})

  -- Set up recording
  if opt.mode == 'eval' and opt.record then
    -- Recreate scratch directory
    paths.rmall('scratch', 'yes')
    paths.mkdir('scratch')

    log.info('Recording screen')
  end

  classic.strict(self)
end


function Display:recordAndDisplay(agent, display, step)
  if qt or self.opt.record then
    -- Extract screen in RGB format for saving images for FFmpeg
    local screen = self.opt.saliency ~= 'none' and self:createSaliencyMap(display, agent) or display
    if qt then
      image.display({image=screen, zoom=self.zoom, win=self.window})
    end
    if self.opt.record then
      image.save(paths.concat('scratch', self.opt.game .. '_' .. string.format('%06d', step) .. '.jpg'), screen)
    end
  end
end


-- Update display
function Display:display(agent, display)
  if not qt then return end

  local screen = self.opt.saliency ~= 'none' and self:createSaliencyMap(display, agent) or display
  image.display({image=screen, zoom=self.zoom, win=self.window})
end


-- Computes saliency map for display
function Display:createSaliencyMap(display, agent) -- TODO: Fix saliency display
  local screen = display:clone()

  -- Use red channel for saliency map
  screen:select(1, 1):copy(agent.saliencyMap)

  return screen
end


function Display:createVideo()
  if not self.opt.record then return end
  log.info('Recorded screen')

  -- Create videos directory
  if not paths.dirp('videos') then
    paths.mkdir('videos')
  end

  -- Use FFmpeg to create a video from the screens
  log.info('Creating video')
  local fps = self.opt.game == 'catch' and 10 or 60
  os.execute('ffmpeg -framerate ' .. fps .. ' -start_number 1 -i scratch/' .. self.opt.game .. '_%06d.jpg -c:v libvpx-vp9 -crf 0 -b:v 0 videos/' .. self.opt.game .. '.webm')
  log.info('Created video')

  -- Clear scratch space
  paths.rmall('scratch', 'yes')
end


return Display
